import random
import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from torch.utils.data import IterableDataset
from datasets import load_dataset
from tqdm import tqdm
import warnings

from typing import List, Optional, Tuple, Union

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

import os
from string import printable
from torch import nn
import transformers
from transformers import (
    BertModel, 
    XLMRobertaModel, 
    AlbertModel, 
    T5EncoderModel, 
    BitsAndBytesConfig, 
    AutoModel,
    LlamaModel)
    
# from InstructorEmbedding import INSTRUCTOR 
# from sentence_transformers import SentenceTransformer
from beir.retrieval import models as beir_models # hs add

from peft import (
    LoraConfig, 
    TaskType, 
    get_peft_model, 
    set_peft_model_state_dict, 
    prepare_model_for_kbit_training,
    PeftModel,
    PeftConfig
)
from peft.tuners.lora import LoraLayer
from accelerate import Accelerator
from accelerate import dispatch_model, infer_auto_device_map

# from src import utils


class BertRetriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
        return_dict=None
    ):
            
        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]

        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

class T5Retriever(T5EncoderModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
        return_dict: Optional[bool] = None,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
        # print("### DEBUG T5 contriever forward - last_hidden:",last_hidden.shape)

        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]

        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)

        return emb

def load_hf(object_class, model_name,load_in_8bit,bnb_config, device_map, args):
    # _max_memory = {i: '40000MB' for i in range(torch.cuda.device_count())} if bnb_config else None 
    _max_memory = {i: '30000MB' for i in range(torch.cuda.device_count())} if bnb_config else None 
    try:
        print("load hf first option")
        obj = object_class.from_pretrained(
            model_name,
            load_in_8bit=load_in_8bit,
            quantization_config = bnb_config,
            local_files_only=False , 
            device_map=device_map, 
            use_cache= not args.use_gradient_checkpointing,
            trust_remote_code=True,
            max_memory =_max_memory, 
            use_flash_attention_2=args.use_flash_attn)
    except:
        obj = object_class.from_pretrained(
            model_name,
            load_in_8bit=load_in_8bit,
            quantization_config = bnb_config,
            local_files_only=False , 
            device_map=device_map, 
            use_cache= not args.use_gradient_checkpointing,
            trust_remote_code=True)
            
    return obj


def create_and_prepare_model(args):
    device_map = None
    bnb_config = None
    load_in_8bit = args.use_8bit_qunatization

    if args.use_4bit_qunatization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_qunatization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )

        if compute_dtype == torch.float16 and args.use_4bit_qunatization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)

    if args.use_4bit_qunatization or args.use_8bit_qunatization:
        device_map =None #'cuda:0' # None # "auto"  # 
 
   
    retriever_model_id = args.model_path
    args.retriever_model_id = retriever_model_id
    if "t5" in retriever_model_id or "T0" in retriever_model_id or "gtr" in retriever_model_id or 'instructor' in retriever_model_id:
        model_class = T5Retriever
    else:
        model_class = BertRetriever


    if os.path.exists(args.model_path):
        pretrained_dict = torch.load(args.model_path+'/checkpoint.pth', map_location="cpu")
        opt = pretrained_dict["opt"]
        if hasattr(opt, "model_path"):
            retriever_model_id = opt.retriever_model_id
        else:
            retriever_model_id = "bert-base-uncased"

        if "t5" in retriever_model_id or "T0" in retriever_model_id or "gtr" in retriever_model_id or 'instructor' in retriever_model_id:
            model_class = T5Retriever
        else:
            model_class = BertRetriever

        # print("retriever_model_id:",retriever_model_id)

        # tokenizer = load_hf(transformers.AutoTokenizer, retriever_model_id)
        # args = load_hf(transformers.AutoConfig, retriever_model_id, bnb_config=bnb_config)

        model = load_hf(
            model_class, 
            retriever_model_id, 
            load_in_8bit,
            bnb_config,
            device_map,
            args)
        
        # print("pretrained_dict:",pretrained_dict['model'])
        pretrained_dict = pretrained_dict["model"]
        # pretrained_dict = pretrained_dict["state_dict"]
        if any("encoder_q." in key for key in pretrained_dict.keys()):  # test if model is defined with moco class
            pretrained_dict = {k.replace("encoder_q.", ""): v for k, v in pretrained_dict.items() if "encoder_q." in k}
        
        if any('model.encoder.base_model' in key for key in pretrained_dict.keys()): # for [inbatch / pl module]
            # pretrained_dict =  {k.replace('model.encoder.base_model.model.','') :p for k,p in pretrained_dict.items()}
            pretrained_dict =  {k.replace('model.encoder.base_model.model.','base_model.model.') :p for k,p in pretrained_dict.items()}
        
        model.load_state_dict(pretrained_dict,strict=False)
        tokenizer = AutoTokenizer.from_pretrained(retriever_model_id, trust_remote_code=True)
        peft_config = None

    else:
        model = load_hf(
            model_class, 
            args.model_path, 
            load_in_8bit,
            bnb_config,
            device_map,
            args)

        peft_config = None
        if args.use_peft:
            peft_config = LoraConfig(
                inference_mode=False,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
                # target_modules=args.lora_target_modules.split(","),
                target_modules=args.lora_target_modules,
            )
            # if (args.use_4bit_qunatization or args.use_8bit_qunatization) and args.use_peft:
            #     model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.use_gradient_checkpointing)

            # if args.use_gradient_checkpointing:
            #     model.gradient_checkpointing_enable()

            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if not hasattr(tokenizer,'pad_token'): # for LLaMa model
        tokenizer.pad_token = tokenizer.eos_token

    # return model, peft_config, tokenizer
    return model, tokenizer, retriever_model_id
