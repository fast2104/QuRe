"""Train/evaluate positive-only QuRe experiments with QuRe baseline settings."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from model import Exp5TextAuxQure, PairwiseKLQure, PositiveQure, VICRegQure, load_qure
from pfm_variants import PromptFlowQure


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--qure-root", required=True, type=Path)
    parser.add_argument("--fashioniq-root", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_local_data_module():
    """Load our data.py without registering it as ``data``.

    QuRe itself imports ``data.utils``.  Giving this local module a distinct
    name prevents it from shadowing that original QuRe package.
    """
    path = Path(__file__).with_name("data.py")
    spec = importlib.util.spec_from_file_location("positive_fashioniq_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_lr(optimizer, epoch, max_epochs, init_lr, min_lr):
    lr = (init_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * epoch / max_epochs)) + min_lr
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def make_epoch_scheduler(optimizer, init_lr, min_lr, max_epochs):
    """The same per-epoch cosine schedule used by Code-ablation."""
    min_ratio = min_lr / init_lr

    def cosine_factor(epoch):
        progress = min(max(epoch, 0) / max(max_epochs, 1), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_factor)


def rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].detach().cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in state["cuda"]])


def parameter_summary(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "trainable_parameter_percent": 100.0 * trainable / total if total else 0.0,
    }


@torch.no_grad()
def evaluate(model, validation_loaders, text_processor, device):
    model.eval()
    metrics = {}
    recalls10, recalls50 = [], []
    spread_sums = dict(pre_flow_spread=0., post_flow_spread=0., feature_spread=0.)
    num_queries = 0
    for dress_type, loaders in validation_loaders.items():
        gallery_features, gallery_names = [], []
        for images, names in tqdm(loaders["gallery"], desc=f"{dress_type}: gallery", leave=False):
            features = model.target_features(images.to(device, non_blocking=True))
            gallery_features.append(features.cpu())
            gallery_names.extend(names)
        gallery_features = torch.cat(gallery_features)

        query_features, target_names = [], []
        for references, captions, targets in tqdm(loaders["queries"], desc=f"{dress_type}: queries", leave=False):
            captions = [text_processor(caption) for caption in captions]
            features, start, banks = model._sample_features(references.to(device, non_blocking=True), captions)
            for key, tensor in zip(spread_sums, (start, banks, features)):
                squared_spread = tensor.float().var(dim=1, unbiased=False).mean()
                spread_sums[key] += squared_spread.item() * len(features)
            num_queries += len(features)
            del start, banks
            query_features.append(features.cpu())
            target_names.extend(targets)
        query_features = torch.cat(query_features).to(device)

        scores = model.score(query_features, gallery_features)
        sorted_indices = scores.argsort(dim=-1, descending=True).numpy()
        names = np.asarray(gallery_names)[sorted_indices]
        labels = names == np.asarray(target_names)[:, None]
        if not np.all(labels.sum(axis=1) == 1):
            raise RuntimeError(f"Expected exactly one FashionIQ target in {dress_type} gallery")
        # Keep the same FashionIQ benchmark fields used by Code-ablation.
        recall10 = float(labels[:, :10].any(axis=1).mean() * 100.0)
        recall50 = float(labels[:, :50].any(axis=1).mean() * 100.0)
        metrics[f"fashionIQ_{dress_type}_recall@10"] = recall10
        metrics[f"fashionIQ_{dress_type}_recall@50"] = recall50
        recalls10.append(recall10)
        recalls50.append(recall50)
    metrics["average_recall_at10"] = float(np.mean(recalls10))
    metrics["average_recall_at50"] = float(np.mean(recalls50))
    metrics["average_recall"] = (
        metrics["average_recall_at10"] + metrics["average_recall_at50"]
    ) / 2.0
    metrics.update({f"val_{key}": math.sqrt(total / num_queries) for key, total in spread_sums.items()})
    return metrics


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step,
                    best_metric, best_epoch, config, metrics):
    """Checkpoint payload compatible in structure with Code-ablation."""
    return {
        "format_version": 1,
        "checkpoint_type": "positive_qure_full",
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": rng_state(),
        "configs": config,
        "metrics": metrics,
    }


def atomic_save_checkpoint(path, **kwargs):
    path = Path(path)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(save_checkpoint(**kwargs), temporary_path)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def load_training_checkpoint(path, model, optimizer, scheduler, scaler):
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        state = torch.load(path, map_location="cpu")
    model.load_state_dict(
        state["model_state_dict"] if "model_state_dict" in state else state["model"]
    )
    optimizer.load_state_dict(
        state["optimizer_state_dict"] if "optimizer_state_dict" in state else state["optimizer"]
    )
    if "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if "scaler_state_dict" in state:
        scaler.load_state_dict(state["scaler_state_dict"])
    restore_rng_state(state.get("rng_state"))
    return state


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    seed_everything(config["seed"])
    run_started = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This QuRe/BLIP-2 training implementation requires CUDA.")

    data_module = load_local_data_module()
    base, vis_processors, txt_processors = load_qure(config["backbone"], device, args.qure_root)
    if config["num_query_tokens"] != base.query_tokens.shape[1]:
        raise ValueError(
            "Config num_query_tokens must match the loaded QuRe checkpoint "
            f"({config['num_query_tokens']} != {base.query_tokens.shape[1]})."
        )
    if config["max_txt_len"] != base.max_txt_len:
        raise ValueError(
            "Config max_txt_len must match the loaded QuRe checkpoint "
            f"({config['max_txt_len']} != {base.max_txt_len})."
        )
    loss_name = config.get("loss", "inbatch_contrastive")
    if loss_name == "inbatch_contrastive":
        model_class = PositiveQure
        main_loss_key = "loss_contrastive"
        loss_metric_keys = ["loss", main_loss_key, "loss_var"]
    elif loss_name == "pairwise_kl_preference":
        model_class = PairwiseKLQure
        main_loss_key = "loss_pairwise_kl"
        loss_metric_keys = ["loss", main_loss_key, "loss_var"]
    elif loss_name == "exp1_text_aux_contrastive":
        model_class = Exp5TextAuxQure
        main_loss_key = "loss_contrastive_reference_text"
        loss_metric_keys = [
            "loss", "loss_contrastive_reference_text", "loss_contrastive_text", "loss_var"
        ]
    elif loss_name == "vicreg":
        model_class = VICRegQure
        main_loss_key = "loss_align"
        loss_metric_keys = ["loss", "loss_align", "loss_var", "loss_cov"]
    else:
        raise ValueError(
            "Unknown loss. Expected 'inbatch_contrastive', "
            "'pairwise_kl_preference', 'exp1_text_aux_contrastive', or 'vicreg'; "
            f"got: {loss_name!r}"
        )
    model_args = (
        base, config["experiment"], config["max_txt_len"], config["use_variance"],
        config["lambda_var"], config["var_gamma"], config["var_epsilon"],
    )
    if loss_name == "exp1_text_aux_contrastive":
        model = model_class(*model_args, lambda_text=config["lambda_text"]).to(device)
    elif loss_name == "vicreg":
        model = model_class(
            *model_args,
            lambda_align=config["lambda_align"],
            lambda_v=config["lambda_v"],
            lambda_c=config["lambda_c"],
            projector_hidden_dim=config["vicreg_projector_hidden_dim"],
            projector_output_dim=config["vicreg_projector_output_dim"],
            queue_size=config["vicreg_queue_size"],
        ).to(device)
    elif config["pfm"]["variant"] != "baseline":
        model = PromptFlowQure(*model_args, pfm_config=config["pfm"]).to(device)
        loss_metric_keys += ["pre_flow_spread", "post_flow_spread", "feature_spread"]
    else:
        model = model_class(*model_args).to(device)
    parameters = parameter_summary(model)
    print(
        "[MODEL] "
        f"total={parameters['total_parameters']:,} "
        f"trainable={parameters['trainable_parameters']:,} "
        f"frozen={parameters['frozen_parameters']:,} "
        f"({parameters['trainable_parameter_percent']:.3f}% trainable)",
        flush=True,
    )

    # QuRe FashionIQDataset itself uses targetpad_transform(1.25, 224), not
    # LAVIS's train transform. Reuse exactly that transform implementation.
    import sys
    if str(args.qure_root.resolve()) not in sys.path:
        sys.path.insert(0, str(args.qure_root.resolve()))
    from data.utils import targetpad_transform
    transform = targetpad_transform(config["target_ratio"], config["image_size"])
    train_loader = data_module.make_train_loader(
        args.fashioniq_root, transform, config["batch_size"], config["num_workers"]
    )
    val_loaders = data_module.make_validation_loaders(
        args.fashioniq_root, transform, config["eval_batch_size"], config["num_workers"]
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["init_lr"], weight_decay=config["weight_decay"],
    )
    scheduler = make_epoch_scheduler(
        optimizer, config["init_lr"], config["min_lr"], config["epochs"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])
    start_epoch, best_score, best_epoch, global_step = 0, -float("inf"), None, 0
    history_path = output_dir / "history.json"
    history = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = []
        except (OSError, json.JSONDecodeError):
            history = []
    if args.resume:
        state = load_training_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        start_epoch = state["epoch"] + 1
        global_step = state.get("global_step", start_epoch * len(train_loader))
        best_score = state.get("best_metric", state.get("metrics", {}).get("average_recall", best_score))
        best_epoch = state.get("best_epoch")

    for epoch in range(start_epoch, config["epochs"]):
        model.epoch = epoch
        model.train()
        lr = optimizer.param_groups[0]["lr"]
        running = {key: 0.0 for key in loss_metric_keys}
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}")
        for step, (references, targets, captions, target_names) in enumerate(progress, start=1):
            captions = [txt_processors["eval"](caption) for caption in captions]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config["amp"]):
                losses = model(
                    references.to(device, non_blocking=True),
                    targets.to(device, non_blocking=True), captions, target_names,
                )
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}, step {step}")
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            if loss_name == "vicreg":
                model.update_vicreg_queue()
            global_step += 1
            for key in running:
                running[key] += float(losses[key].detach())
            progress.set_postfix(loss=running["loss"] / step, lr=lr)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        epoch_metrics = {key: value / len(train_loader) for key, value in running.items()}
        epoch_metrics.update({"lr": lr, "epoch": epoch + 1, "global_step": global_step})
        loss_report = " ".join(
            f"{key}={epoch_metrics[key]:.6f}" for key in loss_metric_keys
        )
        print(f"[EPOCH {epoch + 1}/{config['epochs']}] {loss_report} lr={lr:.3e}", flush=True)
        if (epoch + 1) % config["evaluate_every"] == 0:
            epoch_metrics.update(evaluate(model, val_loaders, txt_processors["eval"], device))
            print(
                f"[VALIDATION EPOCH {epoch + 1}]\n"
                + json.dumps(
                    {key: value for key, value in epoch_metrics.items() if "recall" in key},
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
            model.train()
        history.append({"epoch": epoch + 1, **epoch_metrics})
        selection_metric = epoch_metrics.get("average_recall", -epoch_metrics["loss"])
        is_best = selection_metric > best_score
        if is_best:
            best_score, best_epoch = selection_metric, epoch + 1
        checkpoint_kwargs = dict(
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            epoch=epoch, global_step=global_step, best_metric=best_score,
            best_epoch=best_epoch, config=config, metrics=epoch_metrics,
        )
        atomic_save_checkpoint(output_dir / "last.pt", **checkpoint_kwargs)
        if is_best:
            atomic_save_checkpoint(output_dir / "best.pt", **checkpoint_kwargs)
        (output_dir / "latest_metrics.json").write_text(
            json.dumps(epoch_metrics, indent=2), encoding="utf-8"
        )
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        (output_dir / "run_summary.json").write_text(json.dumps({
            "run_id": output_dir.name,
            "experiment_dir": str(output_dir.resolve()),
            "config": config,
            "parameters": parameters,
            "optimizer": "AdamW",
            "scheduler": "cosine_per_epoch",
            "best_epoch": best_epoch,
            "best_metric": best_score,
            "best_checkpoint": str((output_dir / "best.pt").resolve()),
            "last_epoch": epoch + 1,
            "global_step": global_step,
            "elapsed_seconds": time.time() - run_started,
            "history_file": str(history_path.resolve()),
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
