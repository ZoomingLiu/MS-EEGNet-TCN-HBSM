"""Synthetic engineering checks, not clinical performance validation."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from forecasting.config import ForecastConfig
from forecasting.dataset import ForecastDataset, load_index, split_signature
from forecasting.features import FeatureScaler, minute_power_features
from forecasting.metrics import probability_metrics
from forecasting.models import LongContextForecaster, MinuteTCN
from forecasting.predict import predict
from forecasting.prepare import prepare
from forecasting.train import load_encoder, train


@pytest.fixture(autouse=True)
def single_thread_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config():
    return ForecastConfig(sample_rate=128, channel_names=("C1", "C2"), history_minutes=1,
                          horizon_minutes=2, postictal_seconds=0, filter_warmup_seconds=0,
                          hidden_dim=8, dropout=0, encoder_dropout=0, encoder_chunk_size=12,
                          batch_size=2, epochs=2, patience=2, checkpoint_encoder=False, device="cpu")


def fixture_catalog(tmp_path, config, gap=0):
    records, events = [], []
    for number, split in enumerate(("train", "validation", "test")):
        rng = np.random.default_rng(number)
        signal = rng.standard_normal((2, 12 * 60 * config.sample_rate)).astype(np.float32)
        signal[:, 4 * 60 * config.sample_rate:6 * 60 * config.sample_rate] *= 4
        file = tmp_path / f"{split}.npy"
        np.save(file, signal)
        records.append(dict(subject_id=f"subject_{number}", session_id="session_a", record_id=split,
                            signal_path=file.name, start_sec=gap if split == "test" else 0,
                            sample_rate=config.sample_rate, channel_names=json.dumps(list(config.channel_names)),
                            signal_unit="uV", causal_filter_verified=True, annotation_complete=True, split=split))
        events.append(dict(subject_id=f"subject_{number}", session_id="session_a", onset_sec=360 + (gap if split == "test" else 0), offset_sec=370 + (gap if split == "test" else 0)))
    records_path, events_path = tmp_path / "records.csv", tmp_path / "events.csv"
    pd.DataFrame(records).to_csv(records_path, index=False)
    pd.DataFrame(events).to_csv(events_path, index=False)
    return records_path, events_path


def test_event_labels_and_censoring(tmp_path, config):
    rp, ep = fixture_catalog(tmp_path, config)
    index = prepare(rp, ep, tmp_path / "index.json", config)
    rows = {r["endpoint_sec"]: r for r in index["rows"] if r["split"] == "train"}
    assert rows[240]["target"] == 1  # exact inclusive t+H seizure onset
    assert rows[180]["target"] == 0
    assert 360 not in rows and 420 not in rows  # ongoing seizure / ictal input history
    assert rows[600]["target"] == -1 and not rows[600]["target_observed"]
    assert rows[720]["target"] == -1
    assert index["audit"]["train_catalog_events"] == 1
    assert index["audit"]["train_unique_events_with_positive_endpoints"] == 1
    assert len(ForecastDataset(index, config, "test", observed_only=False)) > len(ForecastDataset(index, config, "test"))
    with pytest.raises(FileExistsError):
        prepare(rp, ep, tmp_path / "index.json", config)


def test_future_gap_and_split_boundary_are_not_negatives(tmp_path, config):
    rp, ep = fixture_catalog(tmp_path, config)
    frame = pd.read_csv(rp)
    original = np.load(tmp_path / "train.npy")
    np.save(tmp_path / "train_a.npy", original[:, :300 * config.sample_rate])
    np.save(tmp_path / "train_b.npy", original[:, 310 * config.sample_rate:])
    a, b = frame.iloc[0].to_dict(), frame.iloc[0].to_dict()
    a.update(signal_path="train_a.npy", record_id="a")
    b.update(signal_path="train_b.npy", record_id="b", start_sec=310)
    pd.DataFrame([a, b, *frame.iloc[1:].to_dict("records")]).to_csv(rp, index=False)
    index = prepare(rp, ep, tmp_path / "gap.json", config)
    rows = {r["endpoint_sec"]: r for r in index["rows"] if r["split"] == "train"}
    assert rows[240]["target"] == -1  # Known later event, but incomplete future observation.
    assert 360 not in rows

    b.update(start_sec=300, split="test")
    # Empty events keep this test about split embargo, not annotations.
    pd.DataFrame([a, b]).to_csv(rp, index=False)
    pd.DataFrame(columns=["subject_id", "session_id", "onset_sec", "offset_sec"]).to_csv(ep, index=False)
    index = prepare(rp, ep, tmp_path / "split.json", config)
    assert next(r for r in index["rows"] if r["endpoint_sec"] == 240)["target"] == -1


def test_unit_and_channel_guards(tmp_path, config):
    rp, ep = fixture_catalog(tmp_path, config)
    frame = pd.read_csv(rp)
    frame.loc[0, "signal_unit"] = "V"
    frame.to_csv(rp, index=False)
    with pytest.raises(ValueError, match="microvolt"):
        prepare(rp, ep, tmp_path / "bad.json", config)


def test_input_geometry_and_mutated_signal_guard(tmp_path, config):
    rp, ep = fixture_catalog(tmp_path, config)
    path = tmp_path / "index.json"
    index = prepare(rp, ep, path, config)
    dataset = ForecastDataset(load_index(path, config), config, "train")
    item = dataset[0]
    assert tuple(item["windows"].shape) == (1, 12, 2, 640)
    assert tuple(item["power"].shape) == (1, 42)
    with pytest.raises(ValueError, match="mismatch"):
        load_index(path, replace(config, horizon_minutes=3))
    index["rows"][0]["segments"][0][2] += 128
    path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="endpoint"):
        load_index(path, config)


def test_power_branch_preserves_scale():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((12, 2, 640)).astype(np.float32)
    original = minute_power_features(x, 128)
    doubled = minute_power_features(2 * x, 128)
    np.testing.assert_allclose((doubled - original)[:14].reshape(2, 7)[:, :2], np.log(2), atol=1e-5)
    np.testing.assert_allclose((doubled - original)[:14].reshape(2, 7)[:, 2:], np.log(4), atol=1e-5)
    np.testing.assert_allclose(original[14:], doubled[14:], atol=1e-5)


def test_tcn_prefix_causality():
    torch.manual_seed(2)
    model = MinuteTCN(8, dropout=0).eval()
    x = torch.randn(2, 8, 15)
    other = x.clone()
    other[:, :, 8:] += 100
    torch.testing.assert_close(model(x)[:, :, :8], model(other)[:, :, :8])
    assert model.receptive_field == 61


def test_full_default_geometry_forward():
    config = ForecastConfig(checkpoint_encoder=False)
    model = LongContextForecaster(config).eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 15, 12, 22, 1280), torch.zeros(1, 15, 462))
    assert logits.shape == (1,) and torch.isfinite(logits).all()


@pytest.mark.parametrize("branches", ["both", "morphology", "power"])
def test_forward_backward(config, branches):
    cfg = replace(config, branches=branches, checkpoint_encoder=True)
    model = LongContextForecaster(cfg)
    logits = model(torch.randn(2, 1, 12, 2, 640), torch.randn(2, 1, 42))
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.])).backward()
    assert model.head[-1].weight.grad.abs().sum() > 0
    if branches != "power":
        assert model.encoder.temporal_block[0].weight.grad.abs().sum() > 0


def test_fixed_token_overfit():
    torch.manual_seed(8)
    model = torch.nn.Sequential(MinuteTCN(8, (1, 2), 0), torch.nn.Conv1d(8, 1, 1))
    x = torch.randn(8, 8, 15)
    y = torch.tensor([0., 1.] * 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(100):
        logits = model(x)[:, 0, -1]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.05
    assert torch.equal((model(x)[:, 0, -1] >= 0).float(), y)


def test_metrics_handle_ties_and_no_events():
    result = probability_metrics([0, 1, 0, 1], [.5] * 4, .5)
    assert result["average_precision"] == .5 and result["roc_auc"] == .5
    assert result["brier_skill_score"] == 0
    assert probability_metrics([0, 0], [.1, .2])["average_precision"] is None
    with pytest.raises(ValueError, match="binary labels"):
        probability_metrics([0, 1.5], [.1, .8])
    with pytest.raises(ValueError, match="Baseline"):
        probability_metrics([0, 1], [.1, .8], float("nan"))


def test_training_checkpoint_and_inference(tmp_path, config):
    rp, ep = fixture_catalog(tmp_path, config)
    path = tmp_path / "index.json"
    index = prepare(rp, ep, path, config)
    checkpoint = train(path, config, tmp_path / "run")
    payload = torch.load(checkpoint, weights_only=True)
    expected = FeatureScaler.fit(ForecastDataset(index, config, "train").power_features(i) for i in range(len(ForecastDataset(index, config, "train"))))
    np.testing.assert_allclose(payload["scaler"]["mean"], expected.mean)
    summary = predict(checkpoint, path, tmp_path / "predictions", device="cpu")
    assert summary["censored_count"] > 0
    assert summary["endpoint_metrics"]["n"] == summary["observed_count"]
    assert (tmp_path / "predictions" / "predictions.csv").is_file()
    # Changing evaluation labels must never change inference scores.
    original_scores = pd.read_csv(tmp_path / "predictions" / "predictions.csv")["risk"].to_numpy()
    changed = json.loads(path.read_text(encoding="utf-8"))
    for row in changed["rows"]:
        if row["split"] == "test" and row["target_observed"]:
            row["target"] = 1 - row["target"]
    alternate_path = tmp_path / "changed_labels.json"
    alternate_path.write_text(json.dumps(changed), encoding="utf-8")
    predict(checkpoint, alternate_path, tmp_path / "predictions_changed_labels", device="cpu")
    np.testing.assert_array_equal(original_scores, pd.read_csv(tmp_path / "predictions_changed_labels" / "predictions.csv")["risk"].to_numpy())
    frozen = LongContextForecaster(replace(config, freeze_encoder=True, encoder_checkpoint=str(checkpoint)))
    load_encoder(frozen, checkpoint, split_signature(index))
    frozen.train()
    assert not frozen.encoder.training
    assert not any(p.requires_grad for p in frozen.encoder.parameters())
    with pytest.raises(ValueError, match="leakage"):
        load_encoder(frozen, checkpoint, "different-fold")
    payload["config"]["branches"] = "power"
    power_checkpoint = tmp_path / "power_only.pt"
    torch.save(payload, power_checkpoint)
    with pytest.raises(ValueError, match="untrained waveform encoder"):
        load_encoder(frozen, power_checkpoint, split_signature(index))


def test_invalid_configuration():
    with pytest.raises(ValueError, match="receptive"):
        ForecastConfig(history_minutes=90)
    with pytest.raises(ValueError, match="random encoder"):
        ForecastConfig(freeze_encoder=True)
