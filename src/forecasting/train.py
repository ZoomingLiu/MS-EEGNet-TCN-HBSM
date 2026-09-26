"""Train the endpoint forecasting task with train-only feature scaling and validation early stopping."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import load_config
from .dataset import ForecastDataset, load_index, split_signature
from .features import FeatureScaler
from .metrics import probability_metrics
from .models import LongContextForecaster


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; choose device: cpu")
    return device


def load_encoder(model, path, signature):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("train_index_signature") != signature:
        raise ValueError("Encoder checkpoint is not from this exact training split/protocol; refusing possible leakage")
    saved = payload["config"]
    if saved["branches"] == "power":
        raise ValueError("A power-only checkpoint has an untrained waveform encoder; do not reuse it as pretrained")
    if (tuple(saved["channel_names"]), saved["sample_rate"], saved["window_seconds"]) != (
            tuple(model.config.channel_names), model.config.sample_rate, model.config.window_seconds):
        raise ValueError("Encoder geometry mismatch")
    state = {key.removeprefix("encoder."): value for key, value in payload["model_state"].items() if key.startswith("encoder.")}
    model.encoder.load_state_dict(state, strict=True)


def run_epoch(model, loader, device, optimizer=None, config=None):
    model.train(optimizer is not None)
    total_loss, count, all_y, all_p = 0.0, 0, [], []
    for batch in loader:
        if not bool(batch["observed"].all()) or not bool(((batch["target"] == 0) | (batch["target"] == 1)).all()):
            raise ValueError("Censored endpoints must not enter supervised loss")
        x, power, target = (batch[k].to(device, non_blocking=True) for k in ("windows", "power", "target"))
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(x, power)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; inspect signal units and feature scaling")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
        total_loss += float(loss.detach()) * len(target)
        count += len(target)
        all_y.extend(target.detach().cpu().tolist())
        all_p.extend(logits.detach().sigmoid().cpu().tolist())
    if not count:
        raise ValueError("No observed endpoints in this split")
    return total_loss / count, np.asarray(all_y, dtype=int), np.asarray(all_p)


def train(index_path, config, output_dir):
    index = load_index(index_path, config)
    training = ForecastDataset(index, config, "train")
    validation = ForecastDataset(index, config, "validation")
    if not len(training) or not len(validation):
        raise ValueError("Need observed training AND validation endpoints; test is never used for early stopping")
    targets = [index["rows"][i]["target"] for i in training.row_ids]
    if set(targets) != {0, 1}:
        raise ValueError("Training needs both event and non-event labels; audit coverage and independent seizures")
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError("Use a new run directory; existing results/checkpoints are never overwritten")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = choose_device(config.device)
    model = LongContextForecaster(config)
    signature = split_signature(index)
    if config.encoder_checkpoint:
        load_encoder(model, config.encoder_checkpoint, signature)
    scaler = FeatureScaler.fit(training.power_features(i) for i in range(len(training)))
    training.scaler = validation.scaler = scaler
    training.signal_cache.clear()
    training.feature_cache.clear()
    model.to(device)
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
    groups = [{"params": head_params, "lr": config.lr_head}]
    if encoder_params and config.branches != "power":
        groups.append({"params": encoder_params, "lr": config.lr_encoder})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(config.seed)
    common = dict(batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(training, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(validation, shuffle=False, **common)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", config.to_dict())
    write_json(out / "scaler.json", scaler.to_dict())
    baseline = float(np.mean(targets))
    history, best_brier, bad_epochs = [], float("inf"), 0
    for epoch in range(config.epochs):
        train_loss, _, _ = run_epoch(model, train_loader, device, optimizer, config)
        val_loss, y, p = run_epoch(model, val_loader, device)
        scores = probability_metrics(y, p, baseline)
        history.append({"epoch": epoch + 1, "train_bce": train_loss, "validation_bce": val_loss, "validation": scores})
        write_json(out / "history.json", history)
        print(json.dumps(history[-1]), flush=True)
        if scores["brier"] < best_brier:
            best_brier, bad_epochs = scores["brier"], 0
            payload = {"schema_version": 1, "config": config.to_dict(), "model_state": model.state_dict(),
                       "scaler": scaler.to_dict(), "train_index_signature": signature, "train_prevalence": baseline,
                       "epoch": epoch + 1, "validation_metrics": scores, "probability_calibrated": False,
                       "torch_version": str(torch.__version__), "numpy_version": str(np.__version__)}
            temporary = out / "best.partial.pt"
            torch.save(payload, temporary)
            temporary.replace(out / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break
    return out / "best.pt"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    print(train(args.index, load_config(args.config), args.output_dir))


if __name__ == "__main__":
    main()
