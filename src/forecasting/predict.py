"""Export aligned endpoint risk scores, including input-valid endpoints with censored outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import ForecastConfig
from .dataset import ForecastDataset, load_index
from .features import FeatureScaler
from .metrics import probability_metrics
from .models import LongContextForecaster
from .train import choose_device, write_json


def predict(checkpoint_path, index_path, output_dir, split="test", device="auto"):
    if split not in {"train", "validation", "test"}:
        raise ValueError("Invalid split")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("Not a long-context forecast checkpoint")
    config = ForecastConfig.from_dict(payload["config"])
    index = load_index(index_path, config)
    scaler = FeatureScaler(**payload["scaler"])
    dataset = ForecastDataset(index, config, split, scaler, observed_only=False)
    if not len(dataset):
        raise ValueError("No input-eligible endpoints in selected split")
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError("Use a new prediction directory")
    device = choose_device(device)
    model = LongContextForecaster(config).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    outputs = []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0):
            logits = model(batch["windows"].to(device), batch["power"].to(device))
            for row_id, logit in zip(batch["row_id"].tolist(), logits.cpu().tolist()):
                row = index["rows"][row_id]
                outputs.append({k: row[k] for k in ("subject_id", "session_id", "record_id", "split", "endpoint_sec",
                                                    "history_start_sec", "forecast_end_sec", "target", "target_observed")})
                outputs[-1].update(row_id=row_id, logit=logit, risk=float(1 / (1 + np.exp(-np.clip(logit, -700, 700)))),
                                   horizon_seconds=config.horizon_minutes * 60, probability_calibrated=False)
    frame = pd.DataFrame(outputs)
    observed = frame[frame["target_observed"]]
    metrics = probability_metrics(observed["target"].to_numpy(), observed["risk"].to_numpy(), payload["train_prevalence"]) if len(observed) else None
    summary = {"split": split, "input_eligible_count": len(frame), "observed_count": len(observed),
               "censored_count": len(frame) - len(observed), "endpoint_metrics": metrics,
               "warning": "Uncalibrated model estimates. Endpoint metrics are not event sensitivity or clinical validation.",
               "protocol": config.protocol()}
    out.mkdir(parents=True, exist_ok=False)
    frame.to_csv(out / "predictions.csv", index=False)
    write_json(out / "metrics.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    print(json.dumps(predict(args.checkpoint, args.index, args.output_dir, args.split, args.device), indent=2))


if __name__ == "__main__":
    main()
