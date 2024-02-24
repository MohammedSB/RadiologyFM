# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import subprocess
from urllib.parse import urlparse

import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch
from torch import nn


logger = logging.getLogger("dinov2")

def load_pretrained_weights(model, pretrained_weights, checkpoint_key, inflation_method=False):
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

    n_slices = model.patch_embed.patch_size[-1]
    n_chans = model.patch_embed.in_chans
    key = "patch_embed.proj.weight"
    emb = state_dict[key]
    logger.info("Embedding shape:", emb.shape, emb.sum())
    if inflation_method == "centering":
        logger.info("Using centering inflation!")
        emb = emb.sum(1, keepdim=True)  # from colored to grayed
        emb = (
            emb.repeat(1, n_chans, 1, 1) / n_chans
        )  # from 1-channel grayed to n-channel grayed
        emb = emb.unsqueeze(4)
        emb = emb.repeat(1, 1, 1, 1, n_slices)  # from 2D to 3D

        center_idx = n_slices // 2
        all_idxs = list(range(n_slices))
        all_idxs.pop(center_idx)
        emb[:, :, :, :, all_idxs] = 0

        print("New embedding shape:", emb.shape, emb.sum())
        state_dict[key] = emb

    ori_num_patches = state_dict["pos_embed"].shape[1] - 1
    cur_num_patches = model.patch_embed.num_patches

    if ori_num_patches != cur_num_patches:
        emb = state_dict["pos_embed"]
        cls_emb = emb[:, 0]
        emb = emb[:, 1:]
        ori_patch_size = int(ori_num_patches**0.5)
        cur_patch_size = int(cur_num_patches**0.5)
        feature_size = emb.shape[-1]
        emb_resize = emb.view(1, ori_patch_size, ori_patch_size, feature_size)
        emb_resize = emb_resize.permute(0, 3, 1, 2)
        emb_new = F.interpolate(emb_resize, (cur_patch_size, cur_patch_size))
        emb_new = emb_new.permute(0, 2, 3, 1)
        emb_new = emb_new.reshape(1, cur_patch_size * cur_patch_size, feature_size)
        emb_new = emb_new.squeeze(0)
        emb_new = torch.cat((emb_new, cls_emb))
        emb_new = emb_new.unsqueeze(0)
        state_dict["pos_embed"] = emb_new

    msg = model.load_state_dict(state_dict, strict=False)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def show_image_from_tensor(image: torch.Tensor):
    image = image.permute(1,2,0).numpy()
    image = image[:, :, 0]
    image = Image.fromarray(np.uint8(image))
    image.show()

def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


class CosineScheduler(object):
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False
