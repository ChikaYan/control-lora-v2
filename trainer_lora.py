#!/usr/bin/env python
# coding=utf-8

import os
import math
import random
import shutil
import logging
import argparse
import importlib
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from diffusers.loaders import LoraLoaderMixin



import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
import gc
from pyhocon import ConfigFactory
import einops

import transformers
from transformers import AutoTokenizer, PretrainedConfig
import torchvision

# import accelerate
# from accelerate import Accelerator
# from accelerate.logging import get_logger
# from accelerate.utils import ProjectConfiguration, set_seed

from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from diffusers.utils.import_utils import is_xformers_available

import utils
from custom_datasets import CustomDataset
# from models.control_lora import ControlLoRAModel
import imageio

from custom_datasets.real_dataset import FaceDataset



if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.3")



def log_validation(tokenizer, text_encoder, vae, text_embed, unet, control_lora, args, device, weight_dtype, controls):
    print("Running validation... ")

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=control_lora,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype, 
        cache_dir=args.cache_dir,
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    image_logs = []

    for i in range(controls.shape[0]):
        control = controls[i].permute([1,2,0]).cpu().detach().numpy()
        control = Image.fromarray((control * 255).astype(np.uint8))

        images = []

        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(
                    image=control, prompt_embeds=text_embed, num_inference_steps=20, generator=generator
                ).images[0]

            images.append(image)

        image_logs.append(
            {"control": control, "images": images}
        )

    return image_logs


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str, cache_dir: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
        cache_dir=cache_dir,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlLoRA training script.")
    parser.add_argument(
        "--conf",
        type=str,
        default="conf/default.conf",
        # required=True,
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--control_lora_model_name_or_path",
        type=str,
        default='lllyasviel/sd-controlnet-openpose',
        help="Path to pretrained control_lora model or model identifier from huggingface.co/models."
        " If not specified control_lora weights are initialized randomly.",
    )
    parser.add_argument(
        "--control_lora_linear_rank",
        type=int,
        default=4,
        help=("The dimension of the Linear Module LoRA update matrices."),
    )
    parser.add_argument(
        "--control_lora_conv2d_rank",
        type=int,
        default=0,
        help=("The dimension of the Conv2d Module LoRA update matrices."),
    )
    parser.add_argument(
        "--use_conditioning_latent", action="store_true", help="Whether or not to use conditioning latent as controlnet image."
    )
    parser.add_argument(
        "--use_same_level_conditioning_latent", action="store_true", help="Whether or not to use conditioning latent with the same tensor size as the unet input."
    )
    parser.add_argument(
        "--use_dora", action="store_true", help="Whether or not to use dora instead of lora."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="control-lora-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--custom_dataset",
        type=str,
        default=None,
        help=(
            "Custom dataset created by yourself."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the control-lora conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the control-lora conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_samples",
        type=int,
        default=0,
        help=(
            "Number of the validation_prompt and validation_image sampled from training dataset while"
            " `--validation_prompt=None` and `--validation_image=None`."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_control_lora",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--conditioning_type_name",
        type=str,
        default=None,
        help="The new type of conditioning.",
    )

    if input_args is not None:
        args, _ = parser.parse_known_args(input_args)
    else:
        args, _ = parser.parse_known_args()

    return args


def save_model_weights(text_embed, controlnet: ControlNetModel, unet: UNet2DConditionModel=None, control_train_mode='none', unet_train_mode='none', save_dir=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    torch.save(text_embed.data, str(save_dir / f"embed-latest.tensor"))

    if control_train_mode == 'full':
        controlnet.save_pretrained(str(save_dir / f"controlnet-latest"))

    if unet_train_mode == 'lora':
        unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        LoraLoaderMixin.save_lora_weights(
            str(save_dir),
            unet_lora_layers=unet_lora_layers_to_save,
        )
    elif unet_train_mode == 'full':
        unet.save_pretrained(str(save_dir / f"unet-latest"))


def load_model_weights(text_embed, controlnet: ControlNetModel, unet: UNet2DConditionModel, control_train_mode='none', unet_train_mode='none', save_dir=None, device='cuda'):
    text_embed.data = torch.load(str(save_dir / f"embed-latest.tensor")).to(device)

    if control_train_mode == 'full':
        load_controlnet = ControlNetModel.from_pretrained(str(save_dir / f"controlnet-latest"))
        controlnet.register_to_config(**load_controlnet.config)
        controlnet.load_state_dict(load_controlnet.state_dict())
        del load_controlnet

    if unet_train_mode == 'lora':
        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(str(save_dir))
        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet, unet_state_dict, adapter_name="default")

        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                print(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

    elif unet_train_mode == 'full':
        load_unet = UNet2DConditionModel.from_pretrained(str(save_dir / f"unet-latest"))
        unet.register_to_config(**load_unet.config)
        unet.load_state_dict(load_unet.state_dict())
        del load_unet


def main(args):
    conf = ConfigFactory.parse_file(args.conf)
    conf['run_name'] = Path(args.conf).stem

    args.train_batch_size = conf.get('train_batch_size', args.train_batch_size)
    args.learning_rate = conf.get('learning_rate', args.learning_rate)
    args.max_train_steps = conf.get('max_train_steps', args.max_train_steps)
    args.resolution = conf.get('resolution', args.resolution)
    args.seed = conf.get('seed', args.seed)
    args.checkpointing_steps = conf.get('checkpointing_steps', args.checkpointing_steps)
    args.pretrained_model_name_or_path = conf.get('pretrained_model_name_or_path', args.pretrained_model_name_or_path)

    UNET_TRAIN_MODE = conf.get('unet_train_mode', 'none') # none, lora, full
    CONTROLNET_TRAIN_MODE = conf.get('controlnet_train_mode', 'none') # none, full

    logging_dir = Path('log') / conf['run_name']
    logging_dir.mkdir(exist_ok=True, parents=True)

    args.output_dir = str(logging_dir)

    device = 'cuda'

    wandb.init(project='diffuser', config=args)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    # If passed along, set the training seed now.
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False, cache_dir=args.cache_dir)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
            cache_dir=args.cache_dir,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision, cache_dir=args.cache_dir)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir)
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, cache_dir=args.cache_dir
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, cache_dir=args.cache_dir)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, cache_dir=args.cache_dir
    )

    controlnet: ControlNetModel
    if args.control_lora_model_name_or_path:
        print("Loading existing control-lora weights")
        controlnet = ControlNetModel.from_pretrained(args.control_lora_model_name_or_path, cache_dir=args.cache_dir)
    else:
        print("Initializing control-lora weights from unet")
        controlnet = ControlNetModel.from_unet(unet)
    
    controlnet.requires_grad_(False)
    controlnet.eval()
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)


    if UNET_TRAIN_MODE == 'lora':
        unet_lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
        )
        unet.add_adapter(unet_lora_config)
        unet.train()
    elif UNET_TRAIN_MODE == 'full':
        unet.requires_grad_(True)
        unet.train()

    if CONTROLNET_TRAIN_MODE == 'full':
        controlnet.requires_grad_(True)
        controlnet.train()


    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if controlnet.dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {controlnet.dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * 1
        )


    optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = [param for param in controlnet.parameters() if param.requires_grad]
    params_to_optimize += [param for param in unet.parameters() if param.requires_grad]
    
    print(f"Unet num of trainable param: {len([param for param in unet.parameters() if param.requires_grad])}")

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    train_dataset = FaceDataset(
        data_folder=conf['dataset.data_folder'],
        subject_name=conf['subject'],
        json_name='flame_params.json',
        # use_background=False,
        load_body_ldmk=False,
        is_eval=False,
        **conf.get_config('dataset.train')
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=conf['train_batch_size'],
                                                    shuffle=True,
                                                    collate_fn=train_dataset.collate_fn,
                                                    num_workers=4,
                                                    )
    
    test_dataset = FaceDataset(
        data_folder=conf['dataset.data_folder'],
        subject_name=conf['subject'],
        json_name='flame_params.json',
        # use_background=False,
        load_body_ldmk=False,
        is_eval=False,
        **conf.get_config('dataset.test')
    )

    # select a few test frames with hands to be logged during training
    dataset_total_len = len(list((Path(conf['dataset.data_folder']) / conf['subject'] / conf['subject'] / 'all' / 'image').glob('*.png')))
    TEST_IDS = np.array(conf['test_log_ids']) - dataset_total_len - conf.get('dataset.test.frame_interval', [0,0])[0]

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=1,
                                                    shuffle=True,
                                                    collate_fn=test_dataset.collate_fn,
                                                    num_workers=4,
                                                    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * 1,
        num_training_steps=args.max_train_steps * 1,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )


    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    controlnet.to(device, dtype=weight_dtype)

    # initialize text_embed
    def compute_text_embeddings(prompt):
        def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
            if tokenizer_max_length is not None:
                max_length = tokenizer_max_length
            else:
                max_length = tokenizer.model_max_length

            text_inputs = tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

            return text_inputs

        def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
            text_input_ids = input_ids.to(text_encoder.device)

            if text_encoder_use_attention_mask:
                attention_mask = attention_mask.to(text_encoder.device)
            else:
                attention_mask = None

            prompt_embeds = text_encoder(
                text_input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )
            prompt_embeds = prompt_embeds[0]

            return prompt_embeds
        
        with torch.no_grad():
            text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None)
            prompt_embeds = encode_prompt(
                text_encoder,
                text_inputs.input_ids,
                text_inputs.attention_mask,
                text_encoder_use_attention_mask=False,
            )
        return prompt_embeds

    init_prompt = "talking man, white background"
    text_embed = torch.nn.Parameter(compute_text_embeddings(init_prompt).detach().requires_grad_(True))


    gc.collect()
    torch.cuda.empty_cache()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * 1 * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save

    # load_model_weights(text_embed, controlnet, unet, CONTROLNET_TRAIN_MODE, UNET_TRAIN_MODE, Path(args.output_dir) / 'ckpt', device)
    if conf.get('resume', False):
        print('Currently resume is not supported, ckpt can only be used for inference!')

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=False,
    )


    image_logs = None
    test_iter = -1

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            (indices, model_input, ground_truth) = batch
            for k in ground_truth:
                if isinstance(ground_truth[k], torch.Tensor):
                    ground_truth[k] = ground_truth[k].to(device)

            control = einops.rearrange(ground_truth['dwpose_im'], 'b h w c -> b c h w')
            B, _, H, W = control.shape
            gt = einops.rearrange(ground_truth['rgb'].reshape([B, H, W, 3]), 'b h w c -> b c h w').float()
            gt = gt * 2. - 1.
            # control needs to be in [0, 1]
            # gt needs to be in [-1, 1]

            # Convert images to latent space
            latents = vae.encode(gt.to(dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            controllora_image = control.to(dtype=weight_dtype)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)


            # Get the text embedding for conditioning
            encoder_hidden_states = text_embed

            down_block_res_samples, mid_block_res_sample = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controllora_image,
                return_dict=False,
            )

            # Predict the noise residual
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[
                    sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
            ).sample


            if conf.get_bool('train_on_img', False):
                ### train with final image ###
                assert noise_scheduler.config.prediction_type == "epsilon"

                def noise_sched_remove_noise(noise_scheduler, noisy_samples, noise, timesteps):
                    alphas_cumprod = noise_scheduler.alphas_cumprod.to(dtype=noisy_samples.dtype)
                    timesteps = timesteps.to(noisy_samples.device)

                    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
                    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                    while len(sqrt_alpha_prod.shape) < len(noisy_samples.shape):
                        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

                    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                    while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_samples.shape):
                        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

                    original_samples = (noisy_samples - sqrt_one_minus_alpha_prod * noise) / sqrt_alpha_prod
                    return original_samples
                
                denoised_latent = noise_sched_remove_noise(noise_scheduler, noisy_latents, model_pred, timesteps)
                # denoised_latent = noise_sched_remove_noise(noise_scheduler, noisy_latents, noise, timesteps)
                pred_img = vae.decode(denoised_latent / vae.config.scaling_factor)[0]
                loss = F.mse_loss(pred_img.float(), gt.float(), reduction="mean")

            else:
                ### train with standard loss ###
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            progress_bar.update(1)
            global_step += 1

            if global_step == 1 or global_step % args.checkpointing_steps == 0:
                save_model_weights(text_embed, controlnet, unet, CONTROLNET_TRAIN_MODE, UNET_TRAIN_MODE, Path(args.output_dir) / 'ckpt')


            if (global_step == 1 or global_step % conf['log_im_every'] == 0):
                # log train
                image_logs = log_validation(
                    tokenizer,
                    text_encoder,
                    vae,
                    text_embed,
                    unet,
                    controlnet,
                    args,
                    device,
                    weight_dtype,
                    control
                )

                for i, im_log in enumerate(image_logs):
                    imageio.imwrite(
                        os.path.join(str(logging_dir), f"train_{global_step:05d}_{i:02d}.png"),
                        np.concatenate([np.asarray(im_log['images'][-1]), np.array(im_log['control']), (((gt[i] + 1.) / 2.).permute([1,2,0]).cpu().numpy() * 255).astype(np.uint8)], axis=1),
                        )

                # log test
                test_iter += 1
                _, _, test_data = test_dataset[TEST_IDS[test_iter % len(TEST_IDS)]]
                for k, v in test_data.items():
                    try:
                        test_data[k] = v.to(device)
                    except:
                        test_data[k] = v

                control = einops.rearrange(test_data['dwpose_im'][None,...], 'b h w c -> b c h w')
                B, _, H, W = control.shape
                gt = einops.rearrange(test_data['rgb'][None,...].reshape([B, H, W, 3]), 'b h w c -> b c h w').float()
                gt = gt * 2. - 1.

                image_logs = log_validation(
                    tokenizer,
                    text_encoder,
                    vae,
                    text_embed,
                    unet,
                    controlnet,
                    args,
                    device,
                    weight_dtype,
                    control
                )

                for i, im_log in enumerate(image_logs):
                    imageio.imwrite(
                        os.path.join(str(logging_dir), f"val_{global_step:05d}_{i:02d}.png"),
                        np.concatenate([np.asarray(im_log['images'][-1]), np.array(im_log['control']), (((gt[i] + 1.) / 2.).permute([1,2,0]).cpu().numpy() * 255).astype(np.uint8)], axis=1),
                        )



            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            wandb.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    
    save_model_weights(text_embed, controlnet, unet, CONTROLNET_TRAIN_MODE, UNET_TRAIN_MODE, Path(args.output_dir) / 'ckpt')
    
    # log test
    for test_i in range(len(test_dataset)):
        _, model_input, test_data = test_dataset[test_i]
        for k, v in test_data.items():
            try:
                test_data[k] = v.to(device)
            except:
                test_data[k] = v

        control = einops.rearrange(test_data['dwpose_im'][None,...], 'b h w c -> b c h w')
        B, _, H, W = control.shape
        gt = einops.rearrange(test_data['rgb'][None,...].reshape([B, H, W, 3]), 'b h w c -> b c h w').float()
        gt = gt * 2. - 1.

        image_logs = log_validation(
            tokenizer,
            text_encoder,
            vae,
            text_embed,
            unet,
            controlnet,
            args,
            device,
            weight_dtype,
            control
        )

        test_log_dir = logging_dir / 'test'
        test_log_dir.mkdir(exist_ok=True, parents=True)

        for i, im_log in enumerate(image_logs):
            imageio.imwrite(
                os.path.join(str(test_log_dir), f"{model_input['img_name'].item()}.png"),
                np.concatenate([np.asarray(im_log['images'][-1]), np.array(im_log['control']), (((gt[i] + 1.) / 2.).permute([1,2,0]).cpu().numpy() * 255).astype(np.uint8)], axis=1),
                )



    # control_lora = control_lora
    # control_lora.save_pretrained(args.output_dir)



if __name__ == "__main__":
    args = parse_args()
    main(args)
