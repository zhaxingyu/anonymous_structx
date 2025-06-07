import json
import pathlib
import pickle
import transformers
import torch
import os
import copy
import wandb
import gc
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from config import *
from transformers import default_data_collator, LlamaTokenizer
from accelerate import DistributedDataParallelKwargs
from accelerate.state import AcceleratorState

import os
from utils import *
#from auxiliary.token import load_dataset
from llama import Transformer, ModelArgs
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs


torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)

import deepspeed

wandb.login(key="ed620b937040a79aa987e59ab8195525bcdd8dca")
def main(args, SEED):
    group = f"{args.dataset}"
    accelerator.init_trackers(project_name=f"{args.project}",
                              init_kwargs={"wandb":
                                               {"tags": [args.dataset, args.model_name],
                                                "group": group,
                                                "name": f"{args.dataset}_EXP{SEED}",
                                                "config": args}
                                           },
                              )

    seed_everything(seed=SEED)
    accelerator.print(args)


    with accelerator.main_process_first():
        tokenizer = LlamaTokenizer.from_pretrained('Llama-2-7b-hf')
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = 'left'

        dataset, split, edge_index = load_dataset[args.dataset]()

        original_dataset = dataset.map(
            preprocess_original_dataset[args.dataset](tokenizer=tokenizer, max_length=original_len[args.dataset]),
            batched=True,
            batch_size=None,
            remove_columns=[i for i in dataset.column_names if i not in ['node_ids']],
            keep_in_memory=True,
            writer_batch_size=10000,
            num_proc=1,
        ).with_format("torch")

        clm_dataset_train = dataset.map(
            preprocess_train_dataset[args.dataset](tokenizer=tokenizer, max_length=instruction_len[args.dataset]),
            batched=True,
            batch_size=None,
            remove_columns=[i for i in dataset.column_names if i not in ['node_ids']],
            keep_in_memory=True,
            writer_batch_size=10000,
            num_proc=1,
        ).with_format("torch")


        clm_dataset_test = dataset.map(
            preprocess_test_dataset[args.dataset](tokenizer=tokenizer, max_length=instruction_len[args.dataset]),
            batched=True,
            batch_size=None,
            remove_columns=[i for i in dataset.column_names if i not in ['node_ids', 'label', 'text_label']],
            keep_in_memory=True,
            writer_batch_size=10000,
            num_proc=1,
        ).with_format("torch")




    accelerator.wait_for_everyone()

    # Step 2: Build Node Classification Dataset
    train_dataset = clm_dataset_train.select(split['train'])
    val_dataset = clm_dataset_train.select(split['valid'])
    val_dataset_eval = clm_dataset_test.select(split['valid'])
    test_dataset = clm_dataset_test.select(split['test'])


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True,
                                               pin_memory=True, shuffle=True, collate_fn=default_data_collator)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True,
                                             shuffle=False, collate_fn=default_data_collator)
    val_loader_eval = torch.utils.data.DataLoader(val_dataset_eval, batch_size=args.batch_size, drop_last=False,
                                                  pin_memory=True, shuffle=False, collate_fn=default_data_collator)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False,
                                              pin_memory=True, shuffle=False, collate_fn=default_data_collator)


    with open(Path(f"{module_path}/{args.model_name}/") / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(w_lora=False,
                                      w_adapter=True,
                                      adapter_layer=8,
                                      adapter_dim=args.adapter_dim,
                                      adapter_len=args.adapter_len,
                                      lora_alpha=16,
                                      lora_r=8,
                                      num_hops=3,
                                      n_mp_layers=args.n_mp_layers,
                                      rrwp=args.rrwp,
                                      n_encoder_layers=args.n_encoder_layers,
                                      n_decoder_layers=args.n_decoder_layers,
                                      adapter_n_heads=args.adapter_n_heads,
                                      task_level=task_level[args.dataset],
                                      **params)


    model_args.vocab_size = tokenizer.vocab_size
    is_zero3 = False
    if accelerator.distributed_type == "DEEPSPEED":
        if AcceleratorState().deepspeed_plugin.zero_stage == 3:
            is_zero3 = True
            accelerator.print("ZeRO-3 is enabled. Using deepspeed.zero.Init() for model instantiation.")

    with deepspeed.zero.Init(enabled=is_zero3):
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        base_model: Transformer = Transformer(params=model_args, edge_index=edge_index,
                                              input_ids=original_dataset['input_ids'],
                                              input_attention_mask=original_dataset['attention_mask'],
                                              )
        torch.set_default_tensor_type(torch.FloatTensor)


    # # 在调用 accelerator.prepare() 之前，在 CPU 上为 base_model 加载权重。
    # # 这样可以避免所有与 GatheredParameters 和 dtype 不匹配相关的问题。
    # ckpt_path = Path(f"{module_path}/{args.model_name}/consolidated.00.pth")
    # accelerator.print(f"Loading checkpoint from {ckpt_path} onto base model BEFORE accelerator.prepare()")
    # # 仅在主进程上执行加载操作，以避免I/O和内存浪费
    # if accelerator.is_main_process:
    #     ckpt_state_dict = torch.load(ckpt_path, map_location="cpu")
    #     base_model.load_state_dict(ckpt_state_dict, strict=False)
    #     # 立即释放 state_dict 占用的内存
    #     del ckpt_state_dict
    #     gc.collect()
    # accelerator.wait_for_everyone()


    accelerator.print(model_args)

    # Step 4 Set Optimizer
    param_adapter, param_lora = base_model.set_trainable_params_new()


    lr_group = {
        'adapter': args.lr,
        'lora': args.lr,
    }

    wd_group = {
        'adapter': args.wd,
        'lora': args.wd,
    }

    accelerator.print(lr_group)
    accelerator.print(wd_group)

    optimizer = torch.optim.AdamW(
        [
            {'params': param_adapter, 'lr': lr_group['adapter'], 'weight_decay': wd_group['adapter']},
            {'params': param_lora, 'lr': lr_group['lora'], 'weight_decay': wd_group['lora']},
        ],
        betas=(0.9, 0.95))

    model, train_loader, val_loader, val_loader_eval, optimizer = accelerator.prepare(base_model, train_loader,
                                                                                      val_loader, val_loader_eval,
                                                                                      optimizer)
    ckpt_path = Path(f"{module_path}/{args.model_name}/consolidated.00.pth")
    accelerator.print(f"Loading checkpoint from {ckpt_path} AFTER accelerator.prepare()")
    # --- 使用 accelerator.unwrap_model(model) ---
    # 这会返回被 DeepSpeed 和 Accelerate 完全初始化后的原始 Transformer 模型
    unwrapped_model = accelerator.unwrap_model(model)
    trainable_params, all_param = unwrapped_model.print_trainable_params()

    # 添加一个保护性检查，防止意外的 ZeroDivisionError
    if all_param > 0:
        accelerator.print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
    else:
        accelerator.print("Trainable parameter count will be available after the first forward pass in ZeRO-3.")

    # ckpt_path = Path(f"{module_path}/{args.model_name}/consolidated.00.pth")
    # accelerator.print(f"Loading checkpoint from {ckpt_path} after accelerator.prepare()")
    #
    # if is_zero3:
    #     # 对于 ZeRO-3，需要使用特殊的加载方式
    #     # accelerator.load_state_dict 无法直接使用，因为它期望一个完整的 state_dict
    #     # 我们需要用 DeepSpeed Engine 的 API 来加载
    #
    #     # 从 accelerator 中获取 deepspeed engine 对象
    #     deepspeed_engine = accelerator.unwrap_model(model)
    #
    #     # DeepSpeedEngine 有一个 load_checkpoint 方法，但它通常用于加载其自己保存的checkpoint
    #     # 从一个普通的 PyTorch checkpoint 加载到 ZeRO-3 模型，最安全的方式是逐参数加载
    #     # 在主进程加载权重，然后分散给所有进程
    #     # 1. 在主进程上加载 checkpoint 到 CPU
    #     if accelerator.is_main_process:
    #         ckpt_state_dict = torch.load(ckpt_path, map_location="cpu")
    #
    #     # 2. 等待主进程加载完成
    #     accelerator.wait_for_everyone()
    #
    #     # 3. 使用 GatheredParameters 上下文管理器
    #     # 这个上下文管理器会临时将所有分片的参数在 rank 0 上聚合
    #     # 让我们可以在这个上下文中像操作一个普通模型一样加载权重
    #     accelerator.print("Forcing model parameters to bfloat16 to ensure dtype consistency for GatheredParameters.")
    #     unwrapped_model.to(torch.bfloat16)
    #     with deepspeed.zero.GatheredParameters(unwrapped_model.parameters(), modifier_rank=0):
    #         if accelerator.is_main_process:
    #             # 只有主进程（modifier_rank=0）持有完整的、未分片的参数，所以只有它需要加载
    #             accelerator.print("Loading state dict on main process...")
    #
    #             # 使用 unwrapped_model 来加载
    #             unwrapped_model.load_state_dict(ckpt_state_dict, strict=False)
    #
    #             accelerator.print("State dict loaded successfully on main process.")
    #
    #     # 4. 再次同步，确保所有进程都等待加载完成
    #     # GatheredParameters 退出时，rank 0 会将更新后的权重自动分散回所有进程
    #     accelerator.wait_for_everyone()
    #     accelerator.print("Weights distributed to all processes.")
    #
    # else:
    #     # 对于非 ZeRO-3 模式 (ZeRO-2, DDP, etc.)
    #     if accelerator.is_main_process:
    #         ckpt_state_dict = torch.load(ckpt_path, map_location="cpu")
    #         unwrapped_model.load_state_dict(ckpt_state_dict, strict=False)
    #     accelerator.wait_for_everyone()

    # Step 5. Training
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss, best_val_acc = float('inf'), -float('inf')


    for epoch in range(args.num_epochs):

        model.train()
        epoch_loss, accum_loss = 0., 0.

        for step, batch in enumerate(train_loader):

            # 与Zero3不兼容
            #with accelerator.accumulate(model):
            optimizer.zero_grad()
            loss = model(**batch)
            accelerator.backward(loss)

            accelerator.clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
            accelerator.clip_grad_norm_(optimizer.param_groups[1]['params'], 0.1)

            if (step + 1) % args.grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], lr_group['adapter'], step / len(train_loader) + epoch,
                                     args)
                adjust_learning_rate(optimizer.param_groups[1], lr_group['lora'], step / len(train_loader) + epoch,
                                     args)

            optimizer.step()
            epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()


            if (step + 1) % args.grad_steps == 0:
                adapter_lr = optimizer.param_groups[0]["lr"]
                lora_lr = optimizer.param_groups[1]["lr"]

                accelerator.log({'Adapter Lr': adapter_lr, 'Lora Lr': lora_lr})
                accelerator.log({'Accum Loss': accum_loss / args.grad_steps})
                accelerator.print(f"Accum Loss: {accum_loss / args.grad_steps}")
                accum_loss = 0.

            progress_bar.update(1)

        accelerator.print(
            f"Epoch: {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
        accelerator.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})


        val_loss = 0.
        samples_seen = 0
        eval_output = []
        model.eval()

        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                loss = model(**batch)
                val_loss += loss.item()

            accelerator.print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss / len(val_loader)}")
            accelerator.log({'Val Loss': val_loss / len(val_loader)})



            for step, batch in enumerate(val_loader_eval):
                kwargs = {}
                kwargs.update(
                    {"node_ids": batch['node_ids'], "input_ids": batch['input_ids'],
                     "attention_mask": batch['attention_mask'], "max_new_tokens": 15})

                generated_tokens = accelerator.unwrap_model(model).generate(**kwargs)
                generated_tokens_gathered = accelerator.gather(generated_tokens).cpu().numpy()

                if accelerator.num_processes > 1:
                    if step == len(val_loader_eval) - 1:
                        generated_tokens_gathered = generated_tokens_gathered[
                                                    : len(val_loader_eval.dataset) - samples_seen]
                    else:
                        samples_seen += len(generated_tokens_gathered)
                eval_output.append(generated_tokens_gathered)



        eval_decode_output = []
        for batch_output in eval_output:
            eval_decode_output.extend(tokenizer.batch_decode(batch_output, skip_special_tokens=False))

        eval_pred = [item.split('</s>')[0] for item in eval_decode_output]
        eval_pred = [item.split('\n\n###\n\n ')[-1] for item in eval_pred]

        eval_label = val_loader_eval.dataset['text_label']
        pred = [_ == f"{eval_label[i]}" for i, _ in enumerate(eval_pred)]
        val_acc = sum(pred) / len(pred)




        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(accelerator.unwrap_model(model).cpu())
            best_epoch = epoch
            model = model.cuda()

        accelerator.print(f'Epoch {epoch} Val Acc {val_acc} Best Val Acc {best_val_acc} Best Epoch {best_epoch}')
        accelerator.log({'val acc': val_acc})

    accelerator.wait_for_everyone()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    accelerator.wait_for_everyone()


    # Step 5. Evaluating


    model, test_loader = accelerator.prepare(best_model, test_loader)

    samples_seen = 0
    eval_output = []
    model.eval()

    progress_bar_test = tqdm(range(len(test_loader)))

    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            kwargs = {}
            kwargs.update(
                {"node_ids": batch['node_ids'], "input_ids": batch['input_ids'],
                 "attention_mask": batch['attention_mask'], "max_new_tokens": 15})

            generated_tokens = accelerator.unwrap_model(model).generate(**kwargs)
            generated_tokens_gathered = accelerator.gather(generated_tokens).cpu().numpy()

            if accelerator.num_processes > 1:
                if step == len(test_loader) - 1:
                    generated_tokens_gathered = generated_tokens_gathered[: len(test_loader.dataset) - samples_seen]
                else:
                    samples_seen += len(generated_tokens_gathered)

            eval_output.append(generated_tokens_gathered)

        progress_bar_test.update(1)

    # Step 6. Post-processing & Evaluating
    if accelerator.is_local_main_process:
#        if hasattr(args, 'save_dir') and args.save_dir and best_model is not None:  # 确保 best_model 被赋值
#            output_dir = Path(args.save_dir)
#            output_dir.mkdir(parents=True, exist_ok=True)
#            model_save_path = output_dir / f"{args.model_name}_seed{SEED}_best_epoch{best_epoch}.pt"  # 使用 args.model_name 和 SEED 区分
#            torch.save(best_model.state_dict(), model_save_path)
#            accelerator.print(f"Best model saved to {model_save_path}")
        eval_decode_output = []
        for batch_output in eval_output:
            eval_decode_output.extend(tokenizer.batch_decode(batch_output, skip_special_tokens=False))

        eval_pred = [item.split('</s>')[0] for item in eval_decode_output]
        eval_pred = [item.split('\n\n###\n\n ')[-1] for item in eval_pred]

        eval_label = test_loader.dataset['text_label']
        pred = [_ == f"{eval_label[i]}" for i, _ in enumerate(eval_pred)]


        acc = sum(pred) / len(pred)

        accelerator.print(f'Test Acc {acc}')
        accelerator.log({'Test Acc': acc})


if __name__ == "__main__":

    args = parse_args_llama()
    for exp, SEED in enumerate(range(args.exp_num)):
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        transformers.logging.set_verbosity_error()
        accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs, init_kwargs],
                                  gradient_accumulation_steps=args.grad_steps)

        main(args, SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        gc.collect()