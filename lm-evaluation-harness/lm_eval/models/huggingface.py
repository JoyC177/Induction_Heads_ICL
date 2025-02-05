import copy
import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
import random
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, DistributedType, find_executable_batch_size
from packaging import version
from peft import PeftModel
from peft import __version__ as PEFT_VERSION
from tqdm import tqdm
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import Collator, stop_sequences_criteria

from einops import rearrange
import math
import warnings
from torch import nn
from transformers.cache_utils import Cache
import functools
from collections import defaultdict
import pickle
from transformers.models.llama.modeling_llama import rotate_half, apply_rotary_pos_emb, repeat_kv
import itertools
import re

eval_logger = utils.eval_logger

STORAGE_DIR = "projects/" # Where mean activations are saved, defined in activation/mean_input.py

def _get_accelerate_args(
    device_map_option: Optional[str] = "auto",
    max_memory_per_gpu: Optional[Union[int, str]] = None,
    max_cpu_memory: Optional[Union[int, str]] = None,
    offload_folder: Optional[str] = "./offload",
) -> dict:
    """Returns the kwargs needed to apply `accelerate` in `AutoModel.from_pretrained`."""
    max_memory = {}
    if max_memory_per_gpu is not None:
        max_memory_per_gpu_map = {
            device_idx: max_memory_per_gpu
            for device_idx in range(torch.cuda.device_count())
        }
        max_memory.update(max_memory_per_gpu_map)
    if max_cpu_memory is not None:
        max_memory["cpu"] = max_cpu_memory

    args = {}
    if max_memory:
        args["max_memory"] = max_memory
    args["device_map"] = device_map_option
    args["offload_folder"] = offload_folder
    return args


@register_model("hf-auto", "hf", "huggingface")
class HFLM(LM):
    """
    An abstracted Huggingface model class. Enables usage with both models of
    `transformers.AutoModelForCausalLM` and `transformers.AutoModelForSeq2SeqLM` classes.

    Supports data-parallel multi-GPU with HF Accelerate.
    """

    AUTO_MODEL_CLASS = None
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: Optional[Union[str, transformers.PreTrainedModel]] = "gpt2",
        token: Optional[str] = None,
        backend: Optional[
            Literal["default", "causal", "seq2seq"]
        ] = "default",  # override whether the model should be treated as decoder-only (causal) or encoder-decoder (seq2seq)
        revision: Optional[str] = "main",
        subfolder: Optional[str] = None,
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ] = None,
        truncation: Optional[bool] = False,
        max_length: Optional[int] = None,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = 64,
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        parallelize: Optional[bool] = False,
        device_map_option: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[Union[str, os.PathLike]] = "./offload",
        # PEFT and quantization options
        peft: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # optionally: take in an already-initialized transformers.PreTrainedModel
        if not isinstance(pretrained, str):
            eval_logger.warning(
                "`pretrained` model kwarg is not of type `str`. Many other model arguments may be ignored. Please do not launch via accelerate or use `parallelize=True` if passing an existing model this way."
            )
            assert not parallelize, "`parallelize=True` is not compatible with passing pre-initialized model to `pretrained`"
            self._model = pretrained
            self._device = self._model.device
            self._config = self._model.config
            gpus = 0

            if tokenizer:
                assert isinstance(
                    tokenizer, transformers.PreTrainedTokenizer
                ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
                self.tokenizer = tokenizer
            else:
                # Get tokenizer
                model_name = self._model.name_or_path
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    use_fast=use_fast_tokenizer,
                )

        else:
            assert isinstance(device, str)
            assert isinstance(pretrained, str)
            assert isinstance(batch_size, (int, str))

            gpus = torch.cuda.device_count()
            accelerator = Accelerator()
            if accelerator.num_processes > 1:
                self.accelerator = accelerator

            if not (parallelize or accelerator.num_processes > 1):
                # use user-passed device
                device_list = set(
                    ["cuda", "cpu"]
                    + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
                    + ["mps", "mps:0"]
                )
                if device and device in device_list:
                    self._device = torch.device(device)
                    eval_logger.info(f"Using device '{device}'")
                    if device in ("mps", "mps:0") and version.parse(
                        torch.__version__
                    ) < version.parse("2.1"):
                        raise RuntimeError(
                            f"mps requires torch >= 2.1. You have {torch.__version__}"
                        )
                else:
                    eval_logger.info("Device not specified")
                    eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                    self._device = (
                        torch.device("cuda")
                        if torch.cuda.is_available()
                        else torch.device("cpu")
                    )
            else:
                #print("0", device)
                if device != "cuda":
                    eval_logger.info(
                        f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                    )
                # TODO: include in warning that `load_in_8bit` etc. affect this too
                self._device = torch.device(device)

            # TODO: update this to be less of a hack once subfolder is fixed in HF
            revision = revision + ("/" + subfolder if subfolder is not None else "")

            self._get_config(
                pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )

        # determine which of 'causal' and 'seq2seq' backends to use
        self._get_backend(
            config=self.config, backend=backend, trust_remote_code=trust_remote_code
        )

        # if we passed `pretrained` as a string, initialize our model now
        if isinstance(pretrained, str):
            self._create_model(
                pretrained=pretrained,
                revision=revision,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                parallelize=parallelize,
                device_map_option=device_map_option,
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                peft=peft,
                autogptq=autogptq,
                **kwargs,
            )

        # access self._model through self.model property outside this method
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.tie_weights()

        if isinstance(pretrained, str) and (gpus >= 1 or str(self.device) == "mps"):
            # TODO: can remove this whole snippet except in the mps case, perhaps?
            if not (parallelize or autogptq or hasattr(self, "accelerator")):
                # place model onto device requested manually,
                # if not using HF Accelerate or device_map
                # or any other option that preloads model onto device
                try:
                    self.model.to(self.device)
                except ValueError:
                    eval_logger.debug(
                        "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                    )

        self._create_tokenizer(
            pretrained,
            tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast_tokenizer=use_fast_tokenizer,
        )

        self.truncation = truncation

        self.vocab_size = self.tokenizer.vocab_size
        # select (or create) a pad token to use
        if self.tokenizer.pad_token:
            pass
        elif self.tokenizer.unk_token:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        elif self.tokenizer.eos_token:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        else:
            if self.config.model_type == "qwen":
                # Qwen's trust_remote_code tokenizer does not allow for adding special tokens
                self.tokenizer.pad_token = "<|endoftext|>"
            elif (
                self.tokenizer.__class__.__name__ == "RWKVWorldTokenizer"
                or self.tokenizer.__class__.__name__ == "Rwkv5Tokenizer"
            ):
                # The RWKV world tokenizer, does not allow for adding special tokens / setting the pad token (which is set as 0)
                # The additional tokenizer name check is needed, as there exists rwkv4 models with neox tokenizer
                # ---
                # Note that the world tokenizer class name, might change in the future for the final huggingface merge
                # https://github.com/huggingface/transformers/pull/26963
                assert self.tokenizer.pad_token_id == 0
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        self._max_length = max_length

        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size

        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)

        if isinstance(pretrained, str):
            # multigpu data-parallel support when launched with accelerate
            if gpus > 1:
                if parallelize:
                    if accelerator.num_processes > 1:
                        raise RuntimeError(
                            "Attempted to use both a HF Accelerate `device_map` and to launch via `accelerate launch`. If this is the case, please either remove `parallelize=True` from --model_args or launch outside of the Accelerate launcher."
                        )
                    else:
                        pass
                elif accelerator.num_processes == 1:
                    # if we aren't launching via accelerate, ditch
                    self._rank = 0
                    self._world_size = 1
                else:
                    if gpus > accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                            "If you would like to use data parallelism, please launch the script "
                            "with 'accelerate launch *script*'. "
                            f"Current run will proceed with {accelerator.num_processes} devices."
                        )
                    assert (
                        accelerator.distributed_type
                        in [
                            DistributedType.FSDP,
                            DistributedType.MULTI_GPU,
                        ]
                    ), "Unsupported distributed type provided. Only DDP and FSDP are supported."
                    if accelerator.distributed_type == DistributedType.FSDP:
                        self._model = accelerator.prepare(self.model)
                    else:
                        self._model = accelerator.prepare_model(
                            self.model, evaluation_mode=True
                        )
                    self._device = torch.device(
                        f"cuda:{accelerator.local_process_index}"
                    )
                    self.accelerator = accelerator

                    if self.accelerator.is_local_main_process:
                        eval_logger.info(f"Using {gpus} devices with data parallelism")

                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
        else:
            # if a PreTrainedModel was passed into HFLM, we forgo distributed setup.
            eval_logger.warning(
                "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _get_backend(
        self,
        config: Union[transformers.PretrainedConfig, transformers.AutoConfig],
        backend: Optional[Literal["default", "causal", "seq2seq"]] = "default",
        trust_remote_code: Optional[bool] = False,
    ) -> None:
        """
        Helper method during initialization.
        Determines the backend ("causal" (decoder-only) or "seq2seq" (encoder-decoder))
        model type to be used.
        """
        assert backend in ["default", "causal", "seq2seq"]

        if backend != "default":
            # if we've settled on non-default backend, use that manually
            if backend == "causal":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
            elif backend == "seq2seq":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM
            eval_logger.info(
                f"Overrode HF model backend type, and using type '{backend}'"
            )
        else:
            # determine and use the default HF backend for this model, based on its config + metadata.
            if (
                getattr(config, "model_type")
                in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
            ):
                # first check if model type is listed under seq2seq models, since some
                # models like MBart are listed in both seq2seq and causal mistakenly in HF transformers.
                # these special cases should be treated as seq2seq models.
                self.AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM
            elif (
                getattr(self.config, "model_type") in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
            ):
                self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
            else:
                if not trust_remote_code:
                    eval_logger.warning(
                        "HF model type is neither marked as CausalLM or Seq2SeqLM. \
                    This is expected if your model requires `trust_remote_code=True` but may be an error otherwise."
                    )
                # if model type is neither in HF transformers causal or seq2seq model registries
                # then we default to AutoModelForCausalLM
                self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

        assert self.AUTO_MODEL_CLASS in [
            transformers.AutoModelForCausalLM,
            transformers.AutoModelForSeq2SeqLM,
        ]
        return None

    def _get_config(
        self,
        pretrained: str,
        revision: str = "main",
        trust_remote_code: bool = False,
    ) -> None:
        self._config = transformers.AutoConfig.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

    def _create_model(
        self,
        pretrained: str,
        token: Optional[str] = None,
        revision: Optional[str] = "main",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: Optional[bool] = False,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        # (accelerate naive PP (device_map) options)
        parallelize: Optional[bool] = False,
        device_map_option: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        # PEFT and quantization options
        peft: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        **kwargs,
    ) -> None:
        """
        Initializes an HF or HF-compatible PreTrainedModel from scratch
        inside HFLM, using the kwargs passed into self.__init__().

        Also handles functionality such as AutoGPTQ usage and PEFT wrapping.

        For future similar extensions to AutoGPTQ that are not core to HF's ecosystem,
        (such as PyTorch models that are nearly, but not quite, fully mirroring
        HF's public interface relied on in this HFLM class)
        please consider subclassing HFLM and overriding this and other methods as needed.
        """

        model_kwargs = kwargs if kwargs else {}

        if parallelize:
            model_kwargs.update(
                _get_accelerate_args(
                    device_map_option,  # TODO: phase out device_map_option?
                    max_memory_per_gpu,
                    max_cpu_memory,
                    offload_folder,
                )
            )
        elif "device_map" not in model_kwargs:
            # set a device_map to initialize model on the right GPU.
            # this is needed because it seems that the default behavior
            # for quantized models now seems to be device_map="auto"
            # which breaks data-parallel mode.
            if hasattr(self, "accelerator"):
                model_kwargs.update(
                    {"device_map": {"": f"cuda:{self.accelerator.local_process_index}"}}
                )
            else:
                model_kwargs.update({"device_map": {"": str(self.device)}})

        if not autogptq:
            if model_kwargs.get("load_in_4bit", None):
                assert (
                    transformers.__version__ >= "4.30.0"
                ), "load_in_4bit requires transformers >= 4.30.0"
            if transformers.__version__ >= "4.30.0":
                if model_kwargs.get("load_in_4bit", None):
                    if model_kwargs.get("bnb_4bit_compute_dtype", None):
                        model_kwargs["bnb_4bit_compute_dtype"] = utils.get_dtype(
                            model_kwargs["bnb_4bit_compute_dtype"]
                        )
            self._model = self.AUTO_MODEL_CLASS.from_pretrained(
                pretrained,
                revision=revision,
                torch_dtype=utils.get_dtype(dtype),
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
        else:
            try:
                from auto_gptq import AutoGPTQForCausalLM
            except ModuleNotFoundError:
                raise Exception(
                    "Tried to load auto_gptq, but auto-gptq is not installed ",
                    "please install auto-gptq via pip install lm-eval[gptq] or pip install -e .[gptq]",
                )

            self._model = AutoGPTQForCausalLM.from_quantized(
                pretrained,
                trust_remote_code=trust_remote_code,
                model_basename=None if autogptq is True else Path(autogptq).stem,
                use_safetensors=True
                if autogptq is True
                else autogptq.endswith(".safetensors"),
                **model_kwargs,
            )

        if peft:
            if model_kwargs.get("load_in_4bit", None):
                assert PEFT_VERSION >= "0.4.0", "load_in_4bit requires peft >= 0.4.0"
            self._model = PeftModel.from_pretrained(
                self._model, peft, revision=revision
            )

        return None

    def _create_tokenizer(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ],
        revision: Optional[str] = "main",
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
    ) -> None:
        """
        Helper method during initialization.

        Create a tokenizer object corresponding to the correct
        tokenizer for value of `pretrained`, or use the pre-initialized tokenizer passed.
        """

        if tokenizer:
            if isinstance(tokenizer, str):
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    tokenizer,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    use_fast=use_fast_tokenizer,
                )
            else:
                assert isinstance(
                    tokenizer, transformers.PreTrainedTokenizer
                ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
                self.tokenizer = tokenizer
        else:
            # Get tokenizer based on 'pretrained'
            if isinstance(pretrained, str):
                model_name = pretrained
            else:
                # get the HF hub name via accessor on model
                model_name = self.model.name_or_path
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name,
                revision=revision,
                trust_remote_code=trust_remote_code,
                use_fast=use_fast_tokenizer,
            )
        return None

    def _detect_batch_size(self, requests=None, pos: int = 0):
        if requests:
            _, context_enc, continuation_enc = requests[pos]
            max_length = len(
                (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1]
            )
            max_context_enc = len(context_enc[-(self.max_length + 1) :])
            max_cont_enc = len(continuation_enc[-(self.max_length + 1) :])
        else:
            max_length = self.max_length

        # if OOM, then halves batch_size and tries again
        @find_executable_batch_size(starting_batch_size=self.max_batch_size)
        def forward_batch(batch_size):
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                length = max(max_context_enc, max_cont_enc)
                batched_conts = torch.ones(
                    (batch_size, length), device=self.device
                ).long()
                test_batch = torch.ones((batch_size, length), device=self.device).long()
                call_kwargs = {
                    "attn_mask": test_batch,
                    "labels": batched_conts,
                }
            else:
                call_kwargs = {}
                test_batch = torch.ones(
                    (batch_size, max_length), device=self.device
                ).long()
            for _ in range(5):
                out = F.log_softmax(self._model_call(test_batch, **call_kwargs), dim=-1)  # noqa: F841

            return batch_size

        batch_size = forward_batch()

        if self.world_size > 1:
            # if multi-GPU, always take minimum over all selected batch sizes
            max_rnk_bs = torch.tensor([batch_size], device=self.device)
            gathered = (
                self.accelerator.gather(max_rnk_bs).cpu().detach().numpy().tolist()
            )
            batch_size = min(gathered)
            utils.clear_torch_cache()
            return batch_size

        utils.clear_torch_cache()
        return batch_size

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        """ """
        if add_special_tokens is None:
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                add_special_tokens = False
            elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                add_special_tokens = True

        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
            add_special_tokens = False
        elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
            add_special_tokens = True

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        if left_truncate_len:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"], encoding["attention_mask"]

    def tok_decode(self, tokens):
        if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
            return self.tokenizer.decode(tokens)
        elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps, attn_mask=None, labels=None):
        """
        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :param attn_mask: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :param labels: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        with torch.no_grad():
            if attn_mask is not None or labels is not None:
                    assert attn_mask is not None and labels is not None
                    assert self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM
                    return self.model(
                        input_ids=inps, attention_mask=attn_mask, labels=labels
                    ).logits
            else:
                    assert self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
                    return self.model(inps).logits
    
    def _model_call_block_att(self, inps, model_type, percentage):
        """
        Applies attention knockout to a given model on a specific task and returns logits.

        Args:
            inps (torch.Tensor): Input tensor of shape [batch, sequence_ctx + sequence_cont] or [batch, sequence_ctx].
            model_type (str): Type of model used ('llama' or 'internlm').
            percentage (float): Percentage of induction heads to perform attention knockout for.

        Returns:
            torch.Tensor: Logits of shape [batch, sequence, vocab] from the model's decoder.
        """

        def wrap_attn_forward(forward_fn, layer_idx, model_, block_config):
            """
            Wraps the attention forward function to inject a custom attention mask.

            This mask first applies a standard lower-triangular (causal) mask, and then
            further blocks attention for specific heads and token pairs as given in
            block_config.

            Args:
                forward_fn (function): The original forward function for the attention layer.
                layer_idx (int): Index of the layer whose attention is being modified.
                model_ (object): The model object.
                block_config (dict): Dictionary mapping layer indices to a list of tuples of
                                    (head_index, source_index, target_index) to block.

            Returns:
                function: The wrapped forward function with custom attention masking.
            """
            @functools.wraps(forward_fn)
            def wrapper_fn(**kwargs):
                # Create a copy of the keyword arguments
                new_kwargs = dict(kwargs)


                hs = kwargs['hidden_states']
                num_tokens = len(inps[0]) # Batch size is 1
                num_heads = model_.config.num_attention_heads

                # Create a lower triangular mask (typical for causal models)
                attn_mask = torch.zeros((1, num_heads, num_tokens, num_tokens))
                attn_mask = torch.tril(torch.ones((1, num_heads, num_tokens, num_tokens), dtype=torch.uint8))

                # Block attention between positions for specific heads
                for head_idx, from_idx, to_idx in block_config[layer_idx]:
                    attn_mask[:, head_idx, from_idx, to_idx] = 0

                # Convert the mask to the expected format
                attn_mask = attn_mask.to(dtype=model_.config.torch_dtype)  # fp16 compatibility
                attn_mask = (1.0 - attn_mask) * torch.finfo(model_.config.torch_dtype).min
                attn_mask = attn_mask.to(hs.device)


                new_kwargs["attention_mask"] = attn_mask

                return forward_fn(**new_kwargs)
                
            return wrapper_fn
    
        def set_block_attn_hooks(_model, block_config, model_type):
            """
            Sets hooks on the model's attention layers to apply the custom attention mask.

            This function goes through each layer (if configured to block
            attention on that layer) and replaces the forward function with a wrapped
            version that applies the attention knockout.

            Args:
                _model: The model instance.
                block_config (dict): Dictionary mapping layer indices to lists of
                                 (head_index, source_index, target_index) tuples.
                model_type (str): Model type (e.g., 'llama').

            Returns:
                  list: List of tuples (layer_index, original_forward_fn) to allow later removal.
            """

            hooks = []
            decoder = _model.model.layers

            for i in range(self.config.num_hidden_layers):
                if i in block_config:
                    if 'llama' in model_type.lower():
                        hook = decoder[i].self_attn.forward
                        decoder[i].self_attn.forward = wrap_attn_forward(decoder[i].self_attn.forward,
                                                                            i, _model, block_config)
                    else:
                        hook = decoder[i].attention.forward
                        decoder[i].attention.forward = wrap_attn_forward(decoder[i].attention.forward,
                                                                            i, _model, block_config)

                    hooks.append((i, hook))

            return hooks

        def remove_wrapper(_model, hooks):
            """
            Removes the hooks that modify the forward pass.
            """
            if 'llama' in model_type.lower():
                for i, hook in hooks:
                    _model.model.layers[i].self_attn.forward = hook
            else:
                for i, hook in hooks:
                    _model.model.layers[i].attention.forward = hook

        def trace_with_attn_block(
            _model,
            inp,
            block_config,
            model_type,   
        ):
            """
            Runs the forward pass of the model with the attention knockout applied.

            This function sets the attention hooks, performs a forward pass, and 
            then removes the hooks.

            Args:
                model: The transformer model.
                inp (torch.Tensor): The input tensor.
                block_config (dict): Configuration for blocking attention.
                model_type (str): Model type string.

            Returns:
                torch.Tensor: The model's logits.
            """
            with torch.no_grad():
                # Set up hooks to block attention
                block_attn_hooks = set_block_attn_hooks(_model, block_config, model_type)

                # Run the forward pass
                pred = _model(inp).logits

                # Remove the hooks
                remove_wrapper(_model, block_attn_hooks)

            return pred
        

        
        def find_all_token_indices(tokenized_list, word):
            """
            Finds all occurrences of a word in a list of tokens.

            This function reconstructs words from tokens (e.g., handling tokens
            that start with a special prefix like '▁') and returns the start and end
            indices of each occurrence.

            Args:
                tokenized_list (list of str): List of token strings.
                word (str): The target word to search for.

            Returns:
                list of tuple: Each tuple is (start_index, end_index) for an occurrence.
            """
            results = []
            word_length = len(word)

            # Iterate over the tokenized list to find all possible starts
            for start in range(len(tokenized_list)):
                reconstructed_word = ''
                current_index = start

                # Reconstruct the word from tokens
                while len(reconstructed_word) < word_length and current_index < len(tokenized_list):
                    token = tokenized_list[current_index]
                    # Remove prefix from the token if present
                    cleaned_token = token[1:] if token.startswith('▁') or token.startswith('Ġ') else token
                    reconstructed_word += cleaned_token
                    current_index += 1

                # Check if the reconstructed word matches the input word
                if reconstructed_word == word:
                    results.append((start, current_index))

                    # Continue searching from the end of the current match
                    start = current_index - 1

            return results
        
        def find_top_scoring_heads(prefix_scores, num_heads, num_heads_per_layer):
            """
            Finds the top-scoring attention heads for ablation.

            Args:
                prefix_scores (torch.Tensor):  Tensor of shape [num_layers, heads_per_layer] with prefix-matching scores.
                num_heads (int): Number of heads to ablate.
                num_heads_per_layer (int): Number of heads per layer.

            Returns:
                dict: Dictionary with layers as keys and list of heads to perform attention knockout for as values.
            """
            flat_tensor = prefix_scores.view(-1)
            _, flat_indices = torch.topk(flat_tensor, num_heads)

            # Convert flat indices to 2D indices
            rows = flat_indices // num_heads_per_layer
            cols = flat_indices % num_heads_per_layer

            # Create the knockout dictionary
            ablation_dict = defaultdict(list)
            for (row, col) in zip(rows, cols):
                ablation_dict[row.item()].append(col.item())

            return ablation_dict
        
        def words_elem_ids(tokenized_list, category_words):
            """
            Finds token ranges to block for object categories (e.g., fruits, animals).

            For each occurrence of a target word in the tokenized sequence, the function identifies a range of 
            subsequent tokens to block. 
            There are two scenarios considered:
            1. If the target word is the second word in a two-word sequence, then only the colon (':') token 
                immediately following is blocked.
            2. If the target word is the first word in a two-word sequence, the function blocks all tokens 
                after the word up to the colon token (which marks the end of the second word).

            Args:
            tokenized_list (list of str): The list of token strings representing the input sequence.
            category_words (list of str): List of words in the category.

        Returns:
            list: A list of range objects indicating token positions to block.
            """
            elem_ids = []
            for word in category_words:
                # Find all occurences of tokens in the same category
                occurrences = find_all_token_indices(tokenized_list, word)

                if not occurrences:
                    continue

                for _, end in occurrences:
                    next_idx = end
                    while True: 
                        if self.tokenizer.decode(inps[0][next_idx]) == ':': # Block attention to colon (target word is the second word in a sequence)
                            break
                        
                        if self.tokenizer.decode(inps[0][next_idx+1]) == ':': # Block attention before colon (target word is the first word in a sequence)
                            break

                        next_idx += 1
                    # Add the range of tokens to block
                    elem_ids.extend([range(end, next_idx+1)])
            return elem_ids

        def char_elem_ids(tokenized_list, category_tokens):
            """
            Finds token ranges to block for single-token elements such as labels or newlines.

            Args:
                tokenized_list (list of str): The list of token strings representing the input sequence.
                category_tokens (list of str):  List of tokens in the category.
            Returns:
                list: A list of ranges (each covering one token) to block.
            """
            elem_ids = []
            for token in category_tokens:
                occurrences = find_all_token_indices(tokenized_list, token)

                for _, end in occurrences:
                    elem_ids.extend([range(end, end+1)]) # Block the next token
            return elem_ids
        
        def col_elem_ids(tokenized_list, category_tokens):
            """
            For each occurrence of a colon token found in the
            tokenized list, this function identifies a range of subsequent tokens to block. The blocking
            starts from the token immediately following the colon and extends until a newline token 
            (decoded as '\n\n') is encountered. The newline token itself is not included in the blocked range.
            If the colon is the last token in the sequence, that occurrence is skipped.

            Args:
                tokenized_list (list of str): The list of token strings representing the input sequence.
                category_tokens (list of str): List of tokens in the category.

            Returns:
                list: A list of range objects, where each range represents token positions to be blocked.
            """
            elem_ids = []
            for token in category_tokens:
                occurrences = find_all_token_indices(tokenized_list, token)

                if not occurrences:
                    continue

                for _, end in occurrences:
                    next_idx = end

                     # If the colon is the last token in the sequence, skip it
                    if next_idx == len(tokenized_list):
                        continue

                    # Scan forward from the token after the colon until a double newline token is encountered
                    while True:
                        if self.tokenizer.decode(inps[0][next_idx]) == '\n\n':
                            break
                        
                        next_idx += 1
                        
                    elem_ids.extend([range(end, next_idx)])
            return elem_ids
        
        with open(f'../induction_scores/prefix_scores/{model_type}/pfx_matching.pkl', 'rb') as file:
            data = pickle.load(file)
        prefix_scores = data['mean']

        n_ablate = math.ceil(self.config.num_attention_heads * self.config.num_hidden_layers * percentage)
        ablation_dict = find_top_scoring_heads(prefix_scores, n_ablate, self.config.num_attention_heads)

        # Define the elements to block attention for
        fruits = [
            "apple", "banana", "cherry", "date", "fig", "grape", "kiwi", "lemon", 
            "mango", "nectarine", "papaya", "peach", "pear", "plum", "raspberry", "strawberry"
        ]

        animals = [
            "cat", "dog", "elephant", "fox", "giraffe", "horse", "kangaroo", "lion", 
            "monkey", "panda", "penguin", "rabbit", "shark", "tiger", "whale", "zebra"
        ]

        furniture = [
            "chair", "table", "sofa", "bed", "desk", "cabinet", "dresser", "stool", 
            "bench", "shelf", "wardrobe", "ottoman", "recliner", "couch", "bookcase", "nightstand"
        ]

        professions = [
            "doctor", "nurse", "teacher", "engineer", "lawyer", "artist", "chef", 
            "farmer", "pilot", "soldier", "scientist", "actor", "writer", "plumber", "carpenter", "electrician"
        ]

        vegetables = [
            "beet", "broccoli", "carrot", "celery", "cucumber", "eggplant", "garlic", "lettuce",
            "onion", "pea", "pepper", "potato", "pumpkin", "radish", "spinach", "tomato"
        ]

        vehicles = [
            "bicycle", "boat", "bus", "car", "helicopter", "minivan", "motorcycle", "plane",
            "scooter", "skateboard", "submarine", "tractor", "train", "trolley", "truck", "yacht"
        ]

        body_parts = [
            "ankle", "arm", "ear", "elbow", "eye", "finger", "foot", "hand",
            "knee", "leg", "mouth", "neck", "nose", "shoulder", "toe", "wrist"
        ]

        instruments = [
            "accordion", "banjo", "cello", "clarinet", "drum", "flute", "guitar", "harmonica",
            "harp", "oboe", "piano", "saxophone", "trombone", "trumpet", "ukulele", "violin"
        ]


        labels = ["Foo", "Bar", "Mur", "Res"]
        colons = [":"]
        newlines = ["\n\n"]


        block_ids = []
        detokenized = self.tokenizer.decode(inps[0]) # Get original input string
        tokenized_list = self.tokenizer.convert_ids_to_tokens(inps[0]) # Get tokenized input


        # Use regex to split the decoded input (while preserving punctuation).
        tokens = re.split(r'(:\s*)|(\n{2,})|(\n)|\s+', detokenized)
        tokens = [token for token in tokens if token] # Remove None/empty strings.
        
        colons_blocked = False
        newlines_blocked = False

        # Iterate over tokens (skipping the first 26 tokens, which are always the task instruction)
        for token in tokens[26:]:
            # Find indices of all occurences of a token in the tokenized input
            token_range = find_all_token_indices(tokenized_list, token)

            if token in fruits:
                elem_ids = words_elem_ids(tokenized_list, fruits)
            elif token in animals:
                elem_ids = words_elem_ids(tokenized_list, animals)
            elif token in furniture:
                elem_ids = words_elem_ids(tokenized_list, furniture)
            elif token in professions:
                elem_ids = words_elem_ids(tokenized_list, professions)
            elif token in vegetables:
                elem_ids = words_elem_ids(tokenized_list, vegetables)
            elif token in vehicles:
                elem_ids = words_elem_ids(tokenized_list, vehicles)
            elif token in body_parts:
                elem_ids = words_elem_ids(tokenized_list, body_parts)
            elif token in instruments:
                elem_ids = words_elem_ids(tokenized_list, instruments)    
            elif token in colons and not colons_blocked:
                elem_ids = col_elem_ids(tokenized_list, colons)
                colons_blocked = True
            elif token in newlines and not newlines_blocked:
                elem_ids = char_elem_ids(tokenized_list, newlines)
                newlines_blocked = True
            elif token in labels:
                elem_ids = char_elem_ids(tokenized_list, labels)
            else:
                continue

            # For every occurrence of the token and for every element range,
            # generate a list of source-target pairs to block attention for
            combinations = []    
            for elem_rng in elem_ids:
                for token_rng in token_range:
                    # Block attention to previous tokens
                    if elem_rng[0] < token_rng[0]:
                        combinations.extend(list(itertools.product(range(token_rng[0], token_rng[1]), elem_rng)))
            block_ids.extend(combinations)
        
        # Build a configuration dictionary for each layer and head
        block_config = defaultdict(list)
        for layer, heads in ablation_dict.items():
            for head in heads:
                block_config[layer].extend([(head, source, stok) for source, stok in block_ids])
        
        # Run the model with the attention knockout applied
        _model = self._model
        with torch.no_grad():
            r = trace_with_attn_block(
                _model, inps, block_config, model_type
                )
        return r
    
    def _model_call_mean_abl(self, inps, model_type, percentage, abl_type, seed):
        """
        Computes model predictions using mean ablation on selected attention heads.

        This function modifies the forward pass of the model's attention layers such that, for a
        subset of heads determined either by top prefix-matching scores ("ind") or via random
        selection ("rnd"), the hidden states are replaced by pre-computed mean activations.
        These mean activations are loaded from disk (from a storage directory) and injected into the 
        attention computation for the selected heads.

        Args:
            inps (torch.Tensor): Input tensor to the model.
            model_type (str): The model type (e.g., 'llama' or 'internlm').
            percentage (float): Fraction of total attention heads to ablate.
            abl_type (str): Type of ablation; "ind" for induction-based (top prefix scores) or "rnd" for random.
            seed (int): Random seed used for reproducibility when using random ablation.

        Returns:
            torch.Tensor: The model's logits computed using the modified (ablated) forward pass.
        """
        def set_block_attn_hooks(_model, block_config, model_type):
            """
            Sets hooks on the model's attention layers to perform mean ablation.

            Args:
                _model: The model.
                block_config (dict): Mapping from layer indices to lists of head indices to ablate.
                model_type (str): The model type string ('llama' or 'internlm').

            Returns:
                list: A list of tuples (layer index, original forward function) to allow hook removal later.
            """

            def wrap_attn_forward(forward_fn, layer_idx, model_, block_config):
                """
                Wraps the attention forward function for a given layer to apply mean ablation.

                Args:
                    forward_fn (function): The original forward function of the attention module.
                    layer_idx (int): The index of the current layer.
                    model_ : The model.
                    block_config (dict): Dictionary mapping layer indices to lists of head indices to ablate.

                Returns:
                    function: A wrapped forward function that applies mean ablation.
                """
                @functools.wraps(forward_fn)
                def wrapper_fn(**kwargs):
                    new_kwargs = {}

                    for (k, v) in kwargs.items():
                        new_kwargs[k] = v

                    change_heads = block_config[layer_idx]
                    
                    # Define two modified forward functions based on the model type.
                    # Each modified forward function loads the mean activations and replaces hidden states
                    # for the specified heads before computing attention.

                    def modified_forward_fn_llama(module, hidden_states, past_key_value=None, attention_mask=None, output_attentions=False, position_ids=None, cache_position=None, use_cache=False):
                        """
                        Modified forward function for Llama models.
                        
                        It loads a pre-computed mean tensor from disk for the given layer, replaces the hidden states 
                        for selected heads, and proceeds with the standard attention computation.
                        """
                        if output_attentions:
                            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
                            # logger.warning_once(
                            #     "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                            #     'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                            # )
                            return super().forward(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_value=past_key_value,
                                output_attentions=output_attentions,
                                use_cache=use_cache,
                                cache_position=cache_position,
                            )

                        bsz, q_len, _ = hidden_states.size()

                        # Create a copy of hidden_states and load the mean tensor
                        hidden_s2 = torch.empty_like(hidden_states).copy_(hidden_states)

                        # Only use the mean values for the current sequence length
                        open_mean = torch.load(f"{STORAGE_DIR}{model_type}/{layer_idx}/mean.pt")
                        hidden_s2[:,:,:] = open_mean[:,:q_len,:]

                        # Compute the standard Q, K, V projections on the original hidden states
                        query_states = module.q_proj(hidden_states)
                        key_states = module.k_proj(hidden_states)
                        value_states = module.v_proj(hidden_states)

                        # Compute the projections on the mean-replaced hidden states
                        query_changed = module.q_proj(hidden_s2)
                        key_changed = module.k_proj(hidden_s2)
                        value_changed = module.v_proj(hidden_s2)

                        query_states = query_states.view(bsz, q_len, module.num_heads, module.head_dim).transpose(1, 2)
                        key_states = key_states.view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)
                        value_states = value_states.view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)

                        query_changed = query_changed.view(bsz, q_len, module.num_heads, module.head_dim).transpose(1, 2)
                        key_changed = key_changed.view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)
                        value_changed = value_changed.view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)

                        cos, sin = module.rotary_emb(value_states, position_ids)
                        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                        cos_changed, sin_changed = module.rotary_emb(value_changed, position_ids)
                        query_changed, key_changed = apply_rotary_pos_emb(query_changed, key_changed, cos_changed, sin_changed)

                        if past_key_value is not None:
                            # sin and cos are specific to RoPE models; cache_position needed for the static cache
                            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                            key_states, value_states = past_key_value.update(key_states, value_states, module.layer_idx, cache_kwargs)

                        key_states = repeat_kv(key_states, module.num_key_value_groups)
                        value_states = repeat_kv(value_states, module.num_key_value_groups)

                        key_changed = repeat_kv(key_changed, module.num_key_value_groups)
                        value_changed = repeat_kv(value_changed, module.num_key_value_groups)

                        # For each head to be ablated, replace the corresponding projections with those computed from the mean tensor
                        for head in change_heads:
                            query_states[:, head, :, :] = query_changed[:, head, :, :]
                            key_states[:, head, :, :] = key_changed[:, head, :, :]
                            value_states[:, head, :, :] = value_changed[:, head, :, :]

                        causal_mask = attention_mask
                        if attention_mask is not None:
                            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

                        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
                        # Reference: https://github.com/pytorch/pytorch/issues/112577.
                        if query_states.device.type == "cuda" and causal_mask is not None:
                            query_states = query_states.contiguous()
                            key_states = key_states.contiguous()
                            value_states = value_states.contiguous()

                        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
                        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
                        is_causal = True if causal_mask is None and q_len > 1 else False

                        attn_output = torch.nn.functional.scaled_dot_product_attention(
                            query_states,
                            key_states,
                            value_states,
                            attn_mask=causal_mask,
                            dropout_p=module.attention_dropout if module.training else 0.0,
                            is_causal=is_causal,
                        )

                        attn_output = attn_output.transpose(1, 2).contiguous()
                        attn_output = attn_output.view(bsz, q_len, -1)

                        attn_output = module.o_proj(attn_output)

                        return attn_output, None, past_key_value
                    
                    def apply_rotary_pos_emb_internlm(q, k, cos, sin, position_ids, unsqueeze_dim=1):
                        """Applies Rotary Position Embedding to the query and key tensors."""
                        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
                        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
                        q_embed = (q * cos) + (rotate_half(q) * sin)
                        k_embed = (k * cos) + (rotate_half(k) * sin)
                        return q_embed, k_embed
                    
                    def modified_forward_fn_internlm(module, hidden_states, past_key_value=None, attention_mask=None, output_attentions=False, position_ids=None, cache_position=None, use_cache=False):
                        """
                        Modified forward function for InternLM2 models.
                        
                        It loads a pre-computed mean tensor from disk for the given layer, replaces the hidden states 
                        for selected heads, and proceeds with the standard attention computation.
                        """
                        bsz, q_len, _ = hidden_states.size()

                        # Create a copy of hidden_states and load the mean tensor
                        hidden_s2 = torch.empty_like(hidden_states).copy_(hidden_states)
                        open_mean = torch.load(f"{STORAGE_DIR}{model_type}/{layer_idx}/mean.pt")

                        # Only use the mean values for the current sequence length
                        hidden_s2[:,:,:] = open_mean[:,:q_len,:]

                        qkv_states = module.wqkv(hidden_states)
                        
                        qkv_states = rearrange(
                            qkv_states,
                            "b q (h gs d) -> b q h gs d",
                            gs=2 + module.num_key_value_groups,
                            d=module.head_dim,
                        )

                        query_states = qkv_states[..., : module.num_key_value_groups, :]
                        query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d")
                        key_states = qkv_states[..., -2, :]
                        value_states = qkv_states[..., -1, :]

                        query_states = query_states.transpose(1, 2)
                        key_states = key_states.transpose(1, 2)
                        value_states = value_states.transpose(1, 2)

                        kv_seq_len = key_states.shape[-2]
                        if past_key_value is not None:
                            # print(past_key_value)
                            kv_seq_len += past_key_value[0].shape[-2]
                        
                        cos, sin = module.rotary_emb(value_states, seq_len=kv_seq_len)
                        query_states, key_states = apply_rotary_pos_emb_internlm(query_states, key_states, cos, sin, position_ids)

                        if past_key_value is not None:
                            # reuse k, v, module_attention
                            key_states = torch.cat([past_key_value[0], key_states], dim=2)
                            value_states = torch.cat([past_key_value[1], value_states], dim=2)

                        past_key_value = (key_states, value_states) if use_cache else None

                        key_states = repeat_kv(key_states, module.num_key_value_groups)
                        value_states = repeat_kv(value_states, module.num_key_value_groups)


                        # Compute the QKV projections from the mean-replaced hidden states
                        transformed_states = module.wqkv(hidden_s2)

                        # Rearrange transformed_states to separate heads and groups
                        transformed_states = rearrange(
                            transformed_states,
                            "b q (h gs d) -> b q h gs d",
                            gs=2 + module.num_key_value_groups,
                            d=module.head_dim,
                        )

                        # Extract and rearrange query states
                        alt_query_states = transformed_states[..., :module.num_key_value_groups, :]
                        alt_query_states = rearrange(alt_query_states, "b q h gs d -> b q (h gs) d")

                       # Extract alternate query states (from the mean-replaced activations)
                        alt_key_states = transformed_states[..., -2, :]
                        alt_value_states = transformed_states[..., -1, :]

                        # Transpose for subsequent operations
                        alt_query_states = alt_query_states.transpose(1, 2)
                        alt_key_states = alt_key_states.transpose(1, 2)
                        alt_value_states = alt_value_states.transpose(1, 2)

                        # Calculate the new sequence length for key and value states
                        new_kv_seq_len = alt_key_states.shape[-2]
                        # if past_key_value2 is not None:
                        #     new_kv_seq_len += past_key_value2[0].shape[-2]
                        # Apply rotary positional embeddings
                        cos_emb, sin_emb = module.rotary_emb(alt_value_states, seq_len=new_kv_seq_len)
                        alt_query_states, alt_key_states = apply_rotary_pos_emb_internlm(alt_query_states, alt_key_states, cos_emb, sin_emb, position_ids)

                        alt_key_states = repeat_kv(alt_key_states, module.num_key_value_groups)
                        alt_value_states = repeat_kv(alt_value_states, module.num_key_value_groups)

                        # Replace selected heads with mean-based projections
                        for head in change_heads:
                            query_states[:, head, :, :] = alt_query_states[:, head, :, :]
                            key_states[:, head, :, :] = alt_key_states[:, head, :, :]
                            value_states[:, head, :, :] = alt_value_states[:, head, :, :]


                        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(module.head_dim)

                        if attn_weights.size() != (bsz, module.num_heads, q_len, kv_seq_len):
                            raise ValueError(
                                f"Attention weights should be of size {(bsz, module.num_heads, q_len, kv_seq_len)}, but is"
                                f" {attn_weights.size()}"
                            )

                        if attention_mask is not None:  # no matter the length, we just slice it
                            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                            attn_weights = attn_weights + causal_mask

                        # upcast attention to fp32
                        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                        # print("attn_weights softmaxed", attn_weights[0, 4, -1, :])
                        attn_output = torch.matmul(attn_weights, value_states)

                        if attn_output.size() != (bsz, module.num_heads, q_len, module.head_dim):
                            raise ValueError(
                                f"`attn_output` should be of size {(bsz, module.num_heads, q_len, module.head_dim)}, but is"
                                f" {attn_output.size()}"
                            )

                        attn_output = attn_output.transpose(1, 2).contiguous()
                        attn_output = attn_output.reshape(bsz, q_len, module.hidden_size)

                        attn_output = module.wo(attn_output)

                        if not output_attentions:
                            attn_weights = None
                        
                        return attn_output, attn_weights, past_key_value
                        
                    # Dispatch to the appropriate modified forward function based on model type.
                    if 'llama' in model_type.lower():
                        return modified_forward_fn_llama(forward_fn, **new_kwargs)
                    if 'internlm' in model_type.lower():
                        return modified_forward_fn_internlm(forward_fn, **new_kwargs)

                return wrapper_fn

            hooks = []
            # print(model_type)
            decoder = _model.model.layers


            for i in range(self.config.num_hidden_layers):
                if i in block_config:
                    if 'llama' in model_type.lower():
                        hook = decoder[i].self_attn.forward
                        decoder[i].self_attn.forward = wrap_attn_forward(decoder[i].self_attn,
                                                                            i, _model, block_config)
                    else:
                        hook = decoder[i].attention.forward
                        decoder[i].attention.forward = wrap_attn_forward(decoder[i].attention,
                                                                            i, _model, block_config)

                    hooks.append((i, hook))

            return hooks

        def remove_wrapper(_model, hooks):
            if 'llama' in model_type.lower():
                for i, hook in hooks:
                    _model.model.layers[i].self_attn.forward = hook
            else:
                for i, hook in hooks:
                    _model.model.layers[i].attention.forward = hook
        
        def trace_with_attn_block(
            _model,
            inp,
            block_config, # A list of (source index, target index) to block
            model_type):
            with torch.no_grad():
                block_attn_hooks = set_block_attn_hooks(_model, block_config, model_type)
                pred = _model(inp).logits
                remove_wrapper(_model, block_attn_hooks)

            return pred
        
        def find_top_scoring_heads(prefix_scores, num_heads, num_heads_per_layer):
            """
            Find the top-scoring heads based on prefix scores.
            :param prefix_scores: Tensor containing prefix scores for each head.
            :param num_heads: Number of top heads to find.
            :param num_heads_per_layer: Number of heads per layer in the model.
            :return: Ablation dictionary with layer indices as keys and lists of head indices to ablate as values.
            """
            # Flatten the tensor
            flat_tensor = prefix_scores.view(-1)

            # Find the top N scores and their indices in the flattened tensor
            _, flat_indices = torch.topk(flat_tensor, num_heads)

            # Convert flat indices to 2D indices
            rows = flat_indices // num_heads_per_layer
            cols = flat_indices % num_heads_per_layer

            # Create the ablation dictionary
            ablation_dict = defaultdict(list)
            for (row, col) in zip(rows, cols):
                ablation_dict[row.item()].append(col.item())

            return ablation_dict

        def generate_random_ablation_dict(num_heads_per_layer, num_heads_to_ablate, seed, prefix_scores):
            """
            Generates a random ablation dictionary where heads are randomly selected following the induction head layer distribution.
    
            Args:
                num_heads_per_layer (int): Number of heads per layer.
                num_heads_to_ablate (int): Total number of heads to ablate.
                seed (int): Random seed.
                prefix_scores (Tensor): Prefix scores tensor.
            
            Returns:
                dict: Dictionary with layer indices as keys and lists of head indices to ablate as values.

            """
            ind_ablation_dict = find_top_scoring_heads(prefix_scores, num_heads_to_ablate, num_heads_per_layer)
            # eligible_mask = prefix_scores < 0.4

            # Create the ablation dictionary
            rnd_ablation_dict = defaultdict(list)

            gen = random.Random(seed)
            for layer in ind_ablation_dict:
                # Generate a list of all possible (layer, head) tuples
                all_possible_heads = [head for head in range(num_heads_per_layer)]# if eligible_mask[layer, head]]
                num_heads_to_select = len(ind_ablation_dict[layer])
                # Randomly select heads to ablate
                randomly_selected_heads = gen.sample(all_possible_heads, num_heads_to_select)
                for head in randomly_selected_heads:
                    rnd_ablation_dict[layer].append(head)
                
            # assert same number of heads are selected
            for layer in rnd_ablation_dict:
                assert len(rnd_ablation_dict[layer]) == len(ind_ablation_dict[layer])

            return rnd_ablation_dict

        with open(f'../induction_scores/prefix_scores/{model_type}/pfx_matching.pkl', 'rb') as file:
            data = pickle.load(file)

        prefix_scores = data['mean']
        n_ablate = math.ceil(self.config.num_attention_heads * self.config.num_hidden_layers * percentage)

        if abl_type == "ind":
            ablation_dict = find_top_scoring_heads(prefix_scores, n_ablate, self.config.num_attention_heads)
        elif abl_type == "rnd":
            ablation_dict = generate_random_ablation_dict(self.config.num_attention_heads, n_ablate, seed, prefix_scores)


        _model = self._model
        with torch.no_grad():
            r = trace_with_attn_block(
                _model, inps, ablation_dict, model_type)
        return r

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)
        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )

    def _select_cont_toks(self, logits, contlen=None, inplen=None):
        if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
            assert (
                contlen and inplen
            ), "Must pass input len and cont. len to select scored logits for causal LM"
            # discard right-padding.
            # also discard the input/context tokens. we'll only score continuations.
            logits = logits[inplen - contlen : inplen]
        elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
            assert (
                contlen and not inplen
            ), "Selecting scored logits for Seq2SeqLM requires only cont. len"
            # only discard right-padding.
            # the logits input to this fn only contain decoder-side tokens.
            logits = logits[:contlen]

        return logits

    def _encode_pair(
        self, context: str, continuation: str
    ) -> Tuple[List[int], List[int]]:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tok_encode(context + continuation, add_special_tokens=False)
        context_enc = self.tok_encode(context, add_special_tokens=False)

        # whole_enc = self.tok_encode(context + continuation)
        # context_enc = self.tok_encode(context, add_special_tokens=False)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def loglikelihood(self, requests: List[Instance], mask_postitions, model_type, patch, percentage, abl_type, seed) -> List[Tuple[float, bool]]:
        print("Mask_postitions is set to", mask_postitions)
        new_reqs = []

        for context, continuation in [req.args for req in requests]:
            if context == "":
                # end of text as context
                context_enc, continuation_enc = (
                    [self.eot_token_id],
                    self.tok_encode(continuation),
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs, mask_postitions, model_type, patch, percentage, abl_type, seed)

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        loglikelihoods = []

        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        for (string,) in tqdm([req.args for req in requests], disable=(self.rank != 0)):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            pad_amnt = 0
            if self.world_size > 1:
                # We pad out the external document-level iterator so the inner iterator doesn't hang
                mytensor = torch.tensor(len(rolling_token_windows), device=self.device)
                gathered = (
                    self.accelerator.gather(mytensor).cpu().detach().numpy().tolist()
                )

                pad_amnt = max(gathered) - gathered[self.rank]
                if pad_amnt > 0:
                    rolling_token_windows += pad_amnt * [rolling_token_windows[0]]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
                disable_tqdm=True,
                override_bs=adaptive_batch_size,
            )

            if (self.world_size > 1) and (pad_amnt > 0):
                string_nll = [x[0] for x in string_nll[:-pad_amnt]]
            else:
                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods

    def _batch_scheduler(self, pos, n_reordered_requests):
        sched = pos // int(len(n_reordered_requests) / self.batch_schedule)
        if sched in self.batch_sizes:
            return self.batch_sizes[sched]
        if (len(self.batch_sizes) > 1) and (
            self.batch_sizes[sched - 1] == self.max_batch_size
        ):
            # if previous batch size is already maximal, skip recomputation
            self.batch_sizes[sched] = self.max_batch_size
            return self.batch_sizes[sched]
        print(
            f"Passed argument batch_size = auto:{self.batch_schedule}. Detecting largest batch size"
        )
        self.batch_sizes[sched] = self._detect_batch_size(n_reordered_requests, pos)
        print(f"Determined largest batch size: {self.batch_sizes[sched]}")
        return self.batch_sizes[sched]

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        mask_positions,
        model_type,
        patch,
        percentage,
        abl_type,
        seed,
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(x):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(requests, sort_fn=_collate)

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(total=len(requests), disable=(disable_tqdm or (self.rank != 0)))
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []

            padding_len_inp = None
            padding_len_cont = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape
                elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                    inp = torch.tensor(
                        (context_enc)[-self.max_length :],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape

                    # build encoder attn masks
                    encoder_attns.append(torch.ones_like(inp))

                    cont = torch.tensor(
                        (continuation_enc)[-self.max_length :],
                        # TODO: left-shift these?
                        # TODO: our code assumes we never end up truncating conts for either model type
                        dtype=torch.long,
                        device=self.device,
                    )
                    (contlen,) = cont.shape

                    conts.append(cont)

                    padding_len_cont = (
                        max(padding_len_cont, contlen)
                        if padding_len_cont is not None
                        else contlen
                    )

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs = {}
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                batched_inps = utils.pad_and_concat(
                    padding_len_inp, inps, padding_side="right"
                )  # [batch, padding_len_inp]
            elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                # TODO: left-pad encoder inps and mask?
                batched_inps = utils.pad_and_concat(
                    padding_len_inp, inps
                )  # [batch, padding_len_inp]
                batched_conts = utils.pad_and_concat(
                    padding_len_cont, conts
                )  # [batch, padding_len_cont]
                batched_encoder_mask = utils.pad_and_concat(
                    padding_len_inp, encoder_attns
                )  # [batch, padding_len_inp]
                call_kwargs = {
                    "attn_mask": batched_encoder_mask,
                    "labels": batched_conts,
                }
            if mask_positions:
                multi_logits = F.log_softmax(
                    self._model_call_block_att(batched_inps, model_type, percentage, **call_kwargs), dim=-1
                )
            elif patch:
                multi_logits = F.log_softmax(
                    self._model_call_mean_abl(batched_inps, model_type, percentage, abl_type, seed, **call_kwargs), dim=-1
                )
            
            else:
                multi_logits = F.log_softmax(
                    self._model_call(batched_inps, **call_kwargs), dim=-1
                )  # [batch, padding_length (inp or cont), vocab]

            for (cache_key, _, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = (
                    inplen + (logits.shape[0] - padding_len_inp)
                    if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
                    else None
                )
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                # print("DEVICE: ", self.device)
                cont_toks = torch.tensor(
                    cont_toks, dtype=torch.long, device=f"cuda:{greedy_tokens.get_device()}"
                ).unsqueeze(0)  # [1, seq]
                max_equal = (greedy_tokens == cont_toks).all()

                # Obtain log-probs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                    -1
                )  # [1, seq]

                # Answer: (log prob, is-exact-match)
                answer = (float(logits.sum()), bool(max_equal))

                res.append(answer)

                self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0))
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size
        # for each different set of kwargs, we execute all requests, by batch.
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else adaptive_batch_size
            if adaptive_batch_size is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and not adaptive_batch_size
            else None
        )

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=batch_size, batch_fn=batch_fn)
        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [kwargs]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            if not until:
                until = [self.tok_decode(self.eot_token_id)]
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            # set the max length in tokens of inputs ("context_enc")
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                # max len for inputs = max length, minus room to generate the max new tokens
                max_ctx_len = self.max_length - max_gen_toks
            elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                # max len for inputs = encoder's whole max_length
                max_ctx_len = self.max_length

            # encode, pad, and truncate contexts for this batch
            context_enc, attn_masks = self.tok_batch_encode(
                contexts,
                left_truncate_len=max_ctx_len,
                truncation=self.truncation,
            )
            context_enc = context_enc.to(self.device)
            attn_masks = attn_masks.to(self.device)

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            # perform batched generation
            cont = self._model_generate(
                context=context_enc,
                attention_mask=attn_masks,
                stop=until,
                **kwargs,
            )

            cont_toks_list = cont.tolist()
            for cont_toks, context in zip(cont_toks_list, contexts):
                # discard context + left-padding toks if using causal decoder-only LM
                if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                    cont_toks = cont_toks[context_enc.shape[1] :]

                s = self.tok_decode(cont_toks)

                # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                for term in until:
                    if len(term) > 0:
                        # ignore '' separator,
                        # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                        s = s.split(term)[0]

                res.append(s)

                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()

        return res