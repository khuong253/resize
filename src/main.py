import json
import logging
import os
import time

import hydra
import torch
from fsspec.core import url_to_fs
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader  
from data.util import compose_transform, sparse_to_dense, split_num_samples
# from fid.model import load_fidnet_v3
from helpers.layout_tokenizer import LayoutPairSequenceTokenizer
from helpers.metric import compute_generative_model_scores
from helpers.sampling import register_sampling_config
from helpers.scheduler import ReduceLROnPlateauWithWarmup
from helpers.util import set_seed
from helpers.visualization import save_image
from hydra_configs import DataConfig, TrainConfig
from models.common.util import load_model, save_model

from crossplatform_util import filter_args_for_ai_platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["HYDRA_FULL_ERROR"] = "1"  # to see full tracelog for hydra

torch.autograd.set_detect_anomaly(True)
total_iter_count = 0


def _to(inputs, device):
    """
    recursively send tensor to the specified device
    """
    outputs = {}
    for k, v in inputs.items():
        if isinstance(v, dict):
            outputs[k] = _to(v, device)
        elif isinstance(v, Tensor):
            outputs[k] = v.to(device)
    return outputs


# if config is not used by hydra.utils.instantiate, define schema to validate args
cs = ConfigStore.instance()
cs.store(group="data", name="base_data_default", node=DataConfig)
cs.store(group="training", name="base_training_default", node=TrainConfig)
register_sampling_config(cs)


@hydra.main(config_path="config", config_name="main", version_base="1.2")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    global total_iter_count
    job_dir = os.path.join(cfg.job_dir, str(cfg.seed))

    fs, _ = url_to_fs(job_dir)
    if not fs.exists(job_dir):
        fs.mkdir(job_dir)
    writer = SummaryWriter(os.path.join(job_dir, "logs"))
    logger.info(cfg)

    if cfg.debug:
        cfg.data.num_workers = 1
        cfg.training.epochs = 2
        cfg.data.batch_size = 64

    with fs.open(os.path.join(job_dir, "config.yaml"), "wb") as file_obj:
        file_obj.write(OmegaConf.to_yaml(cfg).encode("utf-8"))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = compose_transform(cfg.data.transforms)
    train_dataset = instantiate(cfg.dataset)(split="train", transform=transform)
    test_dataset = instantiate(cfg.dataset)(split="test", transform=transform)

    kwargs = {
        "batch_size" : cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_memory": True,
    }
    train_dataloader = DataLoader(train_dataset, shuffle=True, **kwargs)
    val_dataloader = DataLoader(test_dataset, shuffle=False, **kwargs)

    tokenizer = LayoutPairSequenceTokenizer(data_cfg=cfg.data, dataset_cfg=cfg.dataset)

    # LayoutDM
    model = instantiate(cfg.model)(backbone_cfg=cfg.backbone, tokenizer=tokenizer)
    # BART
    # model = instantiate(cfg.model)(backbone_cfg=cfg.backbone, tokenizer=tokenizer, tasks=["c", "cwh", "random"])
    
    optim_groups = model.optim_groups(cfg.training.weight_decay)
    optimizer = instantiate(cfg.optimizer)(optim_groups)

    if cfg.training.pretrained:
        model_path = os.path.join(job_dir, "pretrained-checkpoint.pt")
        model = load_model(model_path, model, device)
        logger.info(f"Loading pretrained model")

    if cfg.training.resume: 
        model_path = os.path.join(job_dir, "checkpoint.pth.tar")
        ckpt = load_model(model_path, model, device)
        start_epoch = int(ckpt["epoch"])
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        logger.info(f"Resuming from checkpoint at epoch {start_epoch} in {job_dir}")

        best_train_ckpt = load_model(os.path.join(job_dir, "best-train-checkpoint.pth.tar"), model, device)
        best_val_ckpt = load_model(os.path.join(job_dir, "best-val-checkpoint.pth.tar"), model, device)
        best_train_loss = float(best_train_ckpt["train_loss"])
        best_val_loss = float(best_val_ckpt["val_loss"])
    else:
        start_epoch = 0
        best_train_loss = float("Inf")
        best_val_loss = float("Inf")

    model = model.to(device)
    scheduler = instantiate(cfg.scheduler)(optimizer=optimizer)
    # fid_model = load_fidnet_v3(train_dataset, cfg.fid_weight_dir, device)

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        model.update_per_epoch(epoch, cfg.training.epochs)

        start_time = time.time()
        train_loss = train(model, train_dataloader, optimizer, cfg, device, writer)
        val_loss = evaluate(model, val_dataloader, cfg, device)
        logger.info(
            "Epoch %d: elapsed = %.1fs:\n- train_loss = %.4f, val_loss = %.4f"
            % (epoch, time.time() - start_time, train_loss, val_loss)
        )
        if any(
            isinstance(scheduler, s)
            for s in [ReduceLROnPlateau, ReduceLROnPlateauWithWarmup]
        ):  
            scheduler.step(val_loss)
        else:
            scheduler.step()

        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("train_loss_epoch_avg", train_loss, epoch)
        writer.add_scalar("val_loss_epoch_avg", val_loss, epoch)

        if epoch % cfg.training.saving_epoch_interval == 0 and epoch != start_epoch:
            save_model(model, job_dir, epoch=epoch)
            logger.info(f"Save current checkpoint at epoch {epoch} in {job_dir}")

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            save_model(model, job_dir, epoch="best_train")
            logger.info(f"Save best train checkpoint in {job_dir}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, job_dir, epoch="best_val")
            logger.info(f"Save best val checkpoint in {job_dir}")

        # if epoch % cfg.training.sample_plot_epoch_interval == 0:
        #     with torch.set_grad_enabled(False):
        #         layouts = model.sample(
        #             batch_size=cfg.data.batch_size,
        #             sampling_cfg=cfg.sampling,
        #             device=device,
        #         )
        #     images = save_image(
        #         layouts["bbox"],
        #         layouts["label"],
        #         layouts["mask"],
        #         val_dataset.colors,
        #     )
        #     tag = f"{cfg.sampling.name}-epoch-{epoch} sampling results"
        #     writer.add_images(tag, images, epoch)

     
def train(
    model: torch.nn.Module,
    train_data: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    device: torch.device,
    writer: SummaryWriter,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    global total_iter_count

    for batch in train_data:
        batch = model.preprocess(batch)
        batch = _to(batch, device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs, losses = model(batch)
            loss = sum(losses.values())
        loss.backward()  # type: ignore

        if cfg.training.grad_norm_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.grad_norm_clip
            )
        
        optimizer.step()
        total_loss += float(loss.item())
        steps += 1
        total_iter_count += 1

        if total_iter_count % cfg.training.loss_plot_iter_interval == 0:
            for (k, v) in losses.items():
                writer.add_scalar(k, v.cpu().item(), total_iter_count + 1)

        # below are for development

        # if cfg.debug:
        #     break

        # if cfg.debug and total_iter_count % 10 == 0:
        #     text = ""
        #     for (k, v) in losses.items():
        #         text += f"{k}: {v} "
        #     print(total_iter_count, text)

        # if cfg.debug and total_iter_count % (cfg.training.loss_plot_iter_interval * 10) == 0:
        #     # sanity check
        #     if cfg.debug:
        #         layouts = model.tokenizer.decode(outputs["outputs"].cpu())
        #         save_image(
        #             layouts["bbox"],
        #             layouts["label"],
        #             layouts["mask"],
        #             train_data.dataset.colors,
        #             f"tmp/debug_{total_iter_count}.png",
        #         )

    return total_loss / steps


def evaluate(
    model: torch.nn.Module,
    test_data: DataLoader,
    cfg: DictConfig,
    device: torch.device,
) -> float:
    total_loss = 0.0
    steps = 0

    model.eval()
    with torch.set_grad_enabled(False):
        for batch in test_data:
            batch = model.preprocess(batch)
            # batch = {k: v.to(device) for (k, v) in batch.items()}
            batch = _to(batch, device)
            _, losses = model(batch)
            loss = sum(losses.values())
            total_loss += float(loss.item())
            steps += 1

            if cfg.debug:
                break

    return total_loss / steps


if __name__ == "__main__":
    filter_args_for_ai_platform()
    main()