#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

from torch.utils.data import Dataset, DataLoader
import argparse
import hashlib
import itertools
import logging
import math
import os
import warnings
from pathlib import Path
from typing import Optional
import random

import accelerate
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import kornia


import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from packaging import version

from torch.distributions import Gamma

from diffusers import StableDiffusionPipeline


from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm


try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


# turns a path into a filename without the extension
def get_filename(path):
    return path.stem


def get_label_from_txt(path):
    txt_path = path.with_suffix(".txt")  # get the path to the .txt file
    if txt_path.exists():
        with open(txt_path, "r") as f:
            return f.read()
    else:
        return ""


def all_images(image_dir):
    image_path = Path(image_dir)
    return (
        list(image_path.glob("*.jpg"))
        + list(image_path.glob("*.png"))
        + list(image_path.glob("*.PNG"))
        + list(image_path.glob("*.webp"))
        + list(image_path.glob("*.jpeg"))
        + list(image_path.glob("*.JPG"))
        + list(image_path.glob("*.JPEG"))
    )


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer_name="M-CLIP/XLM-Roberta-Large-Vit-L-14",
        clip_image_size=224,
        seq_len=77,
        drop_text_prob=0.5,
        drop_image_prob=0.5,
        use_filename_as_label=False,
        use_txt_as_label=False,
        inner_batch_size=None,
        super_image_dir=None,
        super_image_batch_size=4,
    ):
        self.inner_batch_size = inner_batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.transform1 = _transform(clip_image_size)
        self.seq_len = seq_len
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.clip_image_size = clip_image_size

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(
                f"Instance {self.instance_data_root} images root doesn't exists."
            )

        all_files = [f for f in self.instance_data_root.glob('**/*')]
        self.num_instance_images = len(all_files)
        self.instance_prompt = instance_prompt

        self.use_filename_as_label = use_filename_as_label
        self.use_txt_as_label = use_txt_as_label
        self._length = self.num_instance_images

        self.super_image_dir = super_image_dir
        self.super_image_batch_size = super_image_batch_size

        if self.super_image_dir:
            self.super_image_dir = Path(super_image_dir)

        self.aspect_ratio_dirs = sorted(
            list(self.instance_data_root.iterdir()))
        self.folder_weights = [
            len(list(ar_dir.iterdir())) / sum(len(list(ar.iterdir()))
                                              for ar in self.aspect_ratio_dirs)
            for ar_dir in self.aspect_ratio_dirs
        ]

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        aspect_ratio_index = random.choices(
            range(len(self.aspect_ratio_dirs)), weights=self.folder_weights)[0]
        aspect_ratio_dir = self.aspect_ratio_dirs[aspect_ratio_index]
        aspect_ratio_dir_name = Path(aspect_ratio_dir).name
        image_files = list(
            (self.instance_data_root / aspect_ratio_dir_name).iterdir())
        superimage_files = list(
            (self.super_image_dir / aspect_ratio_dir_name).iterdir())

        width, height = map(int, aspect_ratio_dir.name.split("x"))
        batch_size_adjust = (512 * 512) / (width * height)

        image_filepaths = random.sample(image_files, int(
            self.inner_batch_size * batch_size_adjust))
        superimage_filepaths = random.sample(superimage_files, int(
            self.super_image_batch_size * batch_size_adjust))

        size = (width, height)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),

            ]
        )

        images = []
        prompt_ids = []
        masks = []

        for path in image_filepaths + superimage_filepaths:
            abs_path = os.path.abspath(path)
            real_path = os.path.realpath(path)

            import numpy as np

            prompt = (
                get_filename(
                    path) if self.use_filename_as_label else self.instance_prompt
            )
            prompt = get_label_from_txt(
                path) if self.use_txt_as_label else prompt

            # import Fraction
            from fractions import Fraction

            def minimal_aspect_ratio(path):
                filename = Path(path).name
                width, height = map(int, filename.split('x'))
                fraction = Fraction(width, height)
                return f"{fraction.numerator}:{fraction.denominator}"

            aspect = minimal_aspect_ratio(aspect_ratio_dir)
            prompt = prompt + f"; AR: {aspect}"

            if os.path.exists(abs_path):
                try:
                    instance_image = Image.open(abs_path)
                    if not instance_image.mode == "RGB":
                        instance_image = instance_image.convert("RGB")

                    # Text or image random
                    if np.random.binomial(1, self.drop_text_prob):
                        text = ""
                    else:
                        text = prompt

                    text_encoding = self.tokenizer(
                        text,
                        max_length=self.seq_len,
                        padding="max_length",
                        truncation=True,
                        return_attention_mask=True,
                        add_special_tokens=True,
                        return_tensors="pt",
                    )

                    text_ids = text_encoding["input_ids"][0]
                    mask = text_encoding["attention_mask"][0]

                    images.append(self.image_transforms(instance_image))
                    prompt_ids.append(text_ids)
                    masks.append(mask)
                except Exception as e:
                    print(f"Error loading image {abs_path}: {e}")
            else:
                print(f"File not found: {abs_path}")

        example["images"] = images
        example["prompt_ids"] = prompt_ids

        return example


def collate_fn(examples):
    input_ids = [example["prompt_ids"] for example in examples]
    pixel_values = [example["images"] for example in examples]

    import itertools
    input_ids = list(itertools.chain(*input_ids))
    pixel_values = list(itertools.chain(*pixel_values))

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(
        memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    return batch


def create_loader(batch_size, num_workers, shuffle=False, **dataset_params):
    dataset = DreamBoothDataset(**dataset_params)

    return DataLoader(
        dataset,
        # batch_size=batch_size,
        # num_workers=num_workers,
        # shuffle=shuffle,
        pin_memory=True,
    )
