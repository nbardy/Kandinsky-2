import sys
from PIL import Image
import torch
from kandinsky2.model.model_creation import create_model, create_gaussian_diffusion
from kandinsky2.train_utils.train_module_pl2_1 import Decoder
import argparse
import os
from argparse import ArgumentParser
import pytorch_lightning as pl

from kandinsky2.train_utils.data.dataset_unclip_2_1 import create_loader
import kandinsky2.train_utils.data.dataset_unclip_aspect as data_aspect
from kandinsky2.train_utils.utils import freeze_decoder

from kandinsky2.model.text_encoders import TextEncoder
from kandinsky2.model.utils import get_obj_from_str
from kandinsky2.vqgan.autoencoder import VQModelInterface, AutoencoderKL, MOVQ
from kandinsky2.train_utils.trainer_2_1_uclip import train_unclip
from kandinsky2.model.resample import UniformSampler
from omegaconf import OmegaConf
import clip

parser = argparse.ArgumentParser(
    description="Simple example of a training script.")
parser.add_argument(
    "--pretrained_model_name_or_path",
    type=str,
    default=None,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
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
    "--instance_data_dir",
    type=str,
    default=None,
    required=True,
    help="A folder containing the training data of instance images.",
)
parser.add_argument(
    "--class_data_dir",
    type=str,
    default=None,
    required=False,
    help="A folder containing the training data of class images.",
)
parser.add_argument(
    "--instance_prompt",
    type=str,
    default=None,
    required=False,
    help="The prompt with identifier specifying the instance",
)
parser.add_argument(
    "--class_prompt",
    type=str,
    default=None,
    help="The prompt to specify images in the same class as provided instance images.",
)
parser.add_argument(
    "--with_prior_preservation",
    default=False,
    action="store_true",
    help="Flag to add prior preservation loss.",
)
parser.add_argument(
    "--prior_loss_weight",
    type=float,
    default=1.0,
    help="The weight of prior preservation loss.",
)
parser.add_argument(
    "--num_class_images",
    type=int,
    default=100,
    help=(
        "Minimal class images for prior preservation loss. If there are not enough images already present in"
        " class_data_dir, additional images will be sampled with class_prompt."
    ),
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="text-inversion-model",
    help="The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="A seed for reproducible training."
)
parser.add_argument(
    "--train_text_encoder",
    action="store_true",
    help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
)
parser.add_argument(
    "--use_filename_as_label",
    action="store_true",
    help="Uses the filename as the image labels instead of the instance_prompt, useful for regularization when training for styles with wide image variance",
)
parser.add_argument(
    "--use_txt_as_label",
    action="store_true",
    help="Uses the filename.txt file's content as the image labels instead of the instance_prompt, useful for regularization when training for styles with wide image variance",
)
parser.add_argument(
    "--train_batch_size",
    type=int,
    default=4,
    help="Batch size (per device) for the training dataloader.",
)
parser.add_argument(
    "--sample_batch_size",
    type=int,
    default=4,
    help="Batch size (per device) for sampling images.",
)
parser.add_argument("--num_train_epochs", type=int, default=1)
parser.add_argument(
    "--max_train_steps",
    type=int,
    default=None,
    help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
)
parser.add_argument("--super_image_batch_size", type=int, default=4),
parser.add_argument("--inner_batch_size", type=int, default=4),
parser.add_argument("--super_image_dir", type=str, default=None),
parser.add_argument(
    "--checkpointing_steps",
    type=int,
    default=500,
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
    help=(
        "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
        " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
        " for more details"
    ),
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
    "--step_size_up", type=float, default=1000, help="step size up for cyclical LR"
)
parser.add_argument(
    "--step_size_down",
    type=float,
    default=1000,
    help="step size down for cyclical LR",
)
parser.add_argument(
    "--scale_lr",
    action="store_true",
    default=False,
    help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
)
parser.add_argument(
    "--lr_end", type=float, default=1e-9, help="end rate of polynomial lr"
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
    "--lr_warmup_steps",
    type=int,
    default=500,
    help="Number of steps for the warmup in the lr scheduler.",
)
parser.add_argument(
    "--lr_num_cycles",
    type=int,
    default=1,
    help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
)
parser.add_argument(
    "--lr_power",
    type=float,
    default=1.0,
    help="Power factor of the polynomial scheduler.",
)
parser.add_argument(
    "--use_8bit_adam",
    action="store_true",
    help="Whether or not to use 8-bit Adam from bitsandbytes.",
)
parser.add_argument(
    "--stop_text",
    type=int,
    default=999999999,
    help="Stop training the text encoder",
)
parser.add_argument(
    "--dataloader_num_workers",
    type=int,
    default=0,
    help=(
        "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
    ),
)
parser.add_argument(
    "--adam_beta1",
    type=float,
    default=0.9,
    help="The beta1 parameter for the Adam optimizer.",
)
parser.add_argument(
    "--adam_beta2",
    type=float,
    default=0.999,
    help="The beta2 parameter for the Adam optimizer.",
)
parser.add_argument(
    "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
)
parser.add_argument(
    "--adam_epsilon",
    type=float,
    default=1e-08,
    help="Epsilon value for the Adam optimizer",
)
parser.add_argument(
    "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
)
parser.add_argument(
    "--push_to_hub",
    action="store_true",
    help="Whether or not to push the model to the Hub.",
)
parser.add_argument(
    "--hub_token",
    type=str,
    default=None,
    help="The token to use to push to the Model Hub.",
)
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
    "--prior_generation_precision",
    type=str,
    default=None,
    choices=["no", "fp32", "fp16", "bf16"],
    help=(
        "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
        " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
    ),
)
parser.add_argument(
    "--local_rank",
    type=int,
    default=-1,
    help="For distributed training: local_rank",
)
parser.add_argument(
    "--enable_xformers_memory_efficient_attention",
    action="store_true",
    help="Whether or not to use xformers.",
)
parser.add_argument(
    "--enable_xformers_vae", action="store_true", help="add xformers on vae"
)
parser.add_argument(
    "--flash_attention",
    action="store_true",
    help="set memory_effecient_attention to flash attention",
)
parser.add_argument(
    "--channels_last",
    action="store_true",
    help="Whether or not to use channels last.",
)
parser.add_argument(
    "--enable_vae_tiling",
    action="store_true",
    help="Whether or not to use vae tiling.",
)
parser.add_argument(
    "--enable_attention_slicing",
    action="store_true",
    help="Enable Attention Slicing",
)
parser.add_argument(
    "--cpu_model_offload",
    action="store_true",
    help="Enable cpu model offload",
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
parser.add_argument("--save_model_every_n_steps", type=int)
parser.add_argument("--sample_model_every_n_steps", type=int)
parser.add_argument("--pink_noise", type=bool, default=False)
parser.add_argument("--gamma_offset_noise", type=float, default=1.0)
parser.add_argument("--pyramid_noise", type=bool, default=False)
parser.add_argument("--offset_noise", type=bool, default=False)
parser.add_argument("--hi_lo_noise", type=bool, default=False)
parser.add_argument("--gamma_pink_noise", type=bool, default=False)

args = parser.parse_args()


def drop_first_layer(path):
    d = {}
    state_dict = torch.load(path)
    for key in state_dict.keys():
        if key != 'input_blocks.0.0.weight':
            d[key] = state_dict[key]
    return d


def main():
    config = OmegaConf.load('')
    device = config['device']
    model = create_model(**config['model_config'])
    diffusion = create_gaussian_diffusion(**config['diffusion_config'])
    print('start loading')
    if config['params_path'] != '':
        if config['drop_first_layer']:
            model.load_state_dict(drop_first_layer(
                config['params_path']), strict=False)
        else:
            model.load_state_dict(torch.load(config['params_path']))
    model = freeze_decoder(model, **config['freeze']).to(device)
    # train_loader = create_loader(**config['data']['train'])
    # Use new data loader with CLI args
    train_loader = data_aspect.create_loader(args)
    image_encoder = MOVQ(**config['image_enc_params']["params"]).half()
    image_encoder.load_state_dict(torch.load(
        config['image_enc_params']["ckpt_path"]))
    image_encoder = image_encoder.eval().to(device)
    schedule_sampler = UniformSampler(diffusion)
    text_encoder = TextEncoder(
        **config['text_enc_params']).eval().half().to(device)
    optimizer = get_obj_from_str(config['optim_params']["name"])(
        model.parameters(), **config['optim_params']["params"]
    )
    if 'scheduler_params' in config:
        lr_scheduler = get_obj_from_str(config['scheduler_params']["name"])(
            optimizer, **config['scheduler_params']["params"]
        )
    else:
        lr_scheduler = None
    clip_model, _ = clip.load(config['clip_name'], device="cpu", jit=False)
    clip_model.transformer = None
    clip_model.positional_embedding = None
    clip_model.ln_final = None
    clip_model.token_embedding = None
    clip_model.text_projection = None
    clip_model = clip_model.eval().to(device)
    train_unclip(unet=model, diffusion=diffusion, image_encoder=image_encoder,
                 clip_model=clip_model, text_encoder=text_encoder, optimizer=optimizer,
                 lr_scheduler=lr_scheduler, schedule_sampler=schedule_sampler,
                 train_loader=train_loader, val_loader=None, scale=config['image_enc_params']['scale'],
                 num_epochs=config['num_epochs'], save_every=config['save_every'], save_name=config['save_name'],
                 save_path=config['save_path'],  inpainting=config['inpainting'], device=device, args=args)


if __name__ == '__main__':
    main()
