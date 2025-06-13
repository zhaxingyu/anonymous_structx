import os

import torch
from torch import nn
from transformers.modeling_utils import (
    ModuleUtilsMixin, PushToHubMixin,
    logging, Union, Optional, Callable, unwrap_model, get_parameter_dtype,
    FLAX_WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME, WEIGHTS_NAME,
    is_offline_mode, is_remote_url
)
from huggingface_hub import hf_hub_download


logger = logging.get_logger(__name__)


class PushToHubFriendlyModel(nn.Module, ModuleUtilsMixin, PushToHubMixin):
    def __init__(self):
        super().__init__()

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            save_config: bool = True,
            state_dict: Optional[dict] = None,
            save_function: Callable = torch.save,
            push_to_hub: bool = False,
            **kwargs,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

        Arguments:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            save_config (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to save the config of the model. Useful when in distributed training like TPUs and need
                to call this function on all processes. In this case, set :obj:`save_config=True` only on the main
                process to avoid race conditions.
            state_dict (nested dictionary of :obj:`torch.Tensor`):
                The state dictionary of the model to save. Will default to :obj:`self.state_dict()`, but can be used to
                only save parts of the model or if special precautions need to be taken when recovering the state
                dictionary of a model (like when using model parallelism).
            save_function (:obj:`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace :obj:`torch.save` by another method.
            push_to_hub (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to push your model to the Hugging Face model hub after saving it.

                .. warning::

                    Using :obj:`push_to_hub=True` will synchronize the repository you are pushing to with
                    :obj:`save_directory`, which requires :obj:`save_directory` to be a local clone of the repo you are
                    pushing to if it's an existing folder. Pass along :obj:`temp_dir=True` to use a temporary directory
                    instead.

            kwargs:
                Additional key word arguments passed along to the
                :meth:`~transformers.file_utils.PushToHubMixin.push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo = self._create_or_get_repo(save_directory, **kwargs)

        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = unwrap_model(self)

        # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
        # we currently don't use this setting automatically, but may start to use with v5
        dtype = get_parameter_dtype(model_to_save)
        self.pretrain_model.config.torch_dtype = str(dtype).split(".")[1]

        # Attach architecture to the config
        self.pretrain_model.config.architectures = [model_to_save.__class__.__name__]

        # Save the config
        if save_config:
            self.pretrain_model.config.save_pretrained(save_directory)

        # Save the model
        if state_dict is None:
            state_dict = model_to_save.state_dict()

        # Handle the case where some state_dict keys shouldn't be saved
        # if self._keys_to_ignore_on_save is not None:
        #     state_dict = {k: v for k, v in state_dict.items() if k not in self._keys_to_ignore_on_save}

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        save_function(state_dict, output_model_file)

        logger.info(f"Model weights saved in {output_model_file}")

        if push_to_hub:
            url = self._push_to_hub(repo, commit_message=commit_message)
            logger.info(f"Model pushed to the hub in this commit: {url}")

    def load(self, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Adapted for modern huggingface_hub.
        Loads weights from local path or downloads from the Hugging Face Hub.
        """
        # --- 参数解析部分保持不变 ---
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)  # 在新版中，这应该被命名为 token
        revision = kwargs.pop("revision", None)
        # mirror 参数在新版中已废弃

        from_pt = not (from_tf | from_flax)

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        if pretrained_model_name_or_path is None:
            raise ValueError("pretrained_model_name_or_path must be specified")

        # --- 下载逻辑重构 ---
        try:
            # 确定权重文件名
            if from_tf:
                # 还可以根据 TF1/TF2 进一步细化
                filename = TF2_WEIGHTS_NAME
            elif from_flax:
                filename = FLAX_WEIGHTS_NAME
            else:
                filename = WEIGHTS_NAME

            # 使用 hf_hub_download 统一处理
            # 它可以处理本地路径和远程仓库名
            resolved_archive_file = hf_hub_download(
                repo_id=str(pretrained_model_name_or_path),
                filename=filename,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=use_auth_token,
                revision=revision,
            )

        except Exception as err:  # 使用更通用的 Exception，因为 hf_hub_download 抛出的错误类型可能不同
            logger.error(err)
            msg = (
                f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                f"- '{pretrained_model_name_or_path}' is a correct model identifier on 'https://huggingface.co/models'\n\n"
                f"- or '{pretrained_model_name_or_path}' is a path to a directory containing a file named {WEIGHTS_NAME}\n"
            )
            raise EnvironmentError(msg)

        # --- 加载权重到模型的部分保持不变 ---
        if state_dict is None:
            try:
                state_dict = torch.load(resolved_archive_file, map_location="cpu")
            except Exception:
                raise OSError(
                    f"Unable to load weights from PyTorch checkpoint file at '{resolved_archive_file}'"
                )
        self.load_state_dict(state_dict, strict=True)