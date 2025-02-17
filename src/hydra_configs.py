"""
A file to declare dataclass instances used for hydra configs at ./config/*
"""
import os
from dataclasses import dataclass 
from typing import Optional, Tuple, List

from helpers.layout_tokenizer import CHOICES


@dataclass
class TestConfig:
    job_dir: str
    result_dir: str
    dataset_dir: Optional[str] = None  # set if it is different for train/test
    # max_batch_size: int = 512
    max_batch_size: int = 128
    num_run: int = 1  # number of outputs per input
    cond: str = "unconditional"
    num_timesteps: int = 100
    is_validation: bool = False  # for evaluation in validation set (e.g. HP search)
    debug: bool = False  # disable some features to enable fast runtime
    debug_num_samples: int = -1  # in debug mode, reduce the number of samples when > 0

    # for sampling
    sampling: str = "deterministic" # random   # see ./helpers/sampling.py for options
    # below are additional parameters for sampling modes
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: float = 5

    # for unconditional models
    num_uncond_samples: int = 1000

    # for diffusion models
    # assymetric time-difference (https://arxiv.org/abs/2208.04202)
    time_difference: float = 0.0

    # for diffusion models, refinement only
    refine_lambda: float = 3.0  # if > 0.0, trigger refinement mode
    refine_mode: str = "uniform"
    refine_offset_ratio: float = 0.1  # 0.2

    # for diffusion models, relation only
    relation_lambda: float = 3e6  # if > 0.0, trigger relation mode
    relation_mode: str = "average"
    relation_tau: float = 1.0
    relation_num_update: int = 3

    # for continuous diffusion models
    use_ddim: bool = False


@dataclass
class TrainConfig:
    resume_epoch: str = ""
    resume: bool = False 
    pretrained: bool = False 
    epochs: int = 10
    grad_norm_clip: float = 1.0
    weight_decay: float = 1e-1
    saving_epoch_interval: int = 50
    loss_plot_iter_interval: int = 50
    sample_plot_epoch_interval: int = 1
    # fid_plot_num_samples: int = 1000
    # fid_plot_batch_size: int = 512
    fid_plot_num_samples: int = 200
    fid_plot_batch_size: int = 64


@dataclass
class DataConfig:
    batch_size: int = 64 
    bbox_quantization: str = "kmeans"
    num_bin_bboxes: int = 128  
    num_workers: int = 4
    pad_until_max: bool = (
        False  # True for diffusion models, False for others for efficient batching
    )
    shared_bbox_vocab: str = "x-y-w-h"
    special_tokens: Tuple[str] = ("pad", "sep", "mask")
    # special_tokens: Tuple[str] = ("pad",)
    # transforms: Tuple[str] = ("SortByLabel", "LexicographicOrder")
    transforms: Tuple[str] = ()
    var_order: str = "c-x-y-w-h"

    def __post_init__(self) -> None:
        # advanced validation like choices in argparse
        for key in ["shared_bbox_vocab", "bbox_quantization", "var_order"]:
            assert getattr(self, key) in CHOICES[key]