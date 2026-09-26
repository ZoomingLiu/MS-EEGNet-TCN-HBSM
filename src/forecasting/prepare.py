"""Build endpoint labels and history references from a VERIFIED record/event catalog.

No legacy preictal labels, guessed record times, interpolated gaps, or EEG re-filtering.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ForecastConfig, load_config


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _boolean(value, name):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value).lower() in {"true", "1"}:
        return True
    if str(value).lower() in {"false", "0"}:
        return False
    raise ValueError(f"{name} must explicitly be true/false, not {value!r}")


def read_records(path, config):
    path = Path(path).resolve()
    frame = pd.read_csv(path, dtype={"subject_id": str, "session_id": str, "record_id": str})
    required = {"subject_id", "session_id", "record_id", "signal_path", "start_sec", "sample_rate", "channel_names",
                "signal_unit", "causal_filter_verified", "annotation_complete", "split"}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError(f"Record CSV needs complete columns: {sorted(required)}")
    records, paths, identities = [], set(), set()
    for item in frame.to_dict("records"):
        identity = tuple(item[k] for k in ("subject_id", "session_id", "record_id"))
        if identity in identities:
            raise ValueError(f"Duplicate record identity: {identity}")
        identities.add(identity)
        signal_path = (path.parent / str(item["signal_path"])).resolve()
        if signal_path in paths:
            raise ValueError("A signal file cannot occur twice (possible split leakage)")
        paths.add(signal_path)
        if item["split"] not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if item["signal_unit"] != "uV" or not _boolean(item["causal_filter_verified"], "causal_filter_verified"):
            raise ValueError("Verify physical microvolt units and causal filtering before using the forecast pipeline")
        if float(item["sample_rate"]) != config.sample_rate or json.loads(item["channel_names"]) != list(config.channel_names):
            raise ValueError("Sample rate / channel order differs from the configured protocol")
        start = float(item["start_sec"])
        if not np.isfinite(start) or start < 0:
            raise ValueError("start_sec must be a verified nonnegative session-relative time")
        signal = np.load(signal_path, mmap_mode="r", allow_pickle=False)
        if signal.ndim != 2 or signal.shape[0] != len(config.channel_names) or signal.shape[1] == 0 or signal.dtype.kind != "f":
            raise ValueError(f"Expected a floating-point [channels,samples] signal: {signal_path}")
        n_samples = signal.shape[1]
        stat = signal_path.stat()
        del signal
        records.append(dict(subject_id=item["subject_id"], session_id=item["session_id"], record_id=item["record_id"],
                            signal_path=str(signal_path), start_sec=start, end_sec=start + n_samples / config.sample_rate,
                            n_samples=n_samples, file_size=stat.st_size, mtime_ns=stat.st_mtime_ns, split=item["split"],
                            annotation_complete=_boolean(item["annotation_complete"], "annotation_complete")))
    if not records:
        raise ValueError("Empty record catalog")
    records.sort(key=lambda r: (r["subject_id"], r["session_id"], r["start_sec"]))
    for left, right in zip(records, records[1:]):
        if (left["subject_id"], left["session_id"]) == (right["subject_id"], right["session_id"]) and right["start_sec"] < left["end_sec"] - 1e-6:
            raise ValueError("Overlapping records in one session: verify timestamps instead of guessing order")
    return records


def read_events(path, records):
    events = pd.read_csv(path, dtype={"subject_id": str, "session_id": str})
    required = {"subject_id", "session_id", "onset_sec", "offset_sec"}
    if not required.issubset(events.columns) or events[list(required)].isna().any().any():
        raise ValueError(f"Event CSV needs columns {sorted(required)}; use a header-only CSV for no events")
    result = []
    for e in events.to_dict("records"):
        start, end = float(e["onset_sec"]), float(e["offset_sec"])
        if not np.isfinite([start, end]).all() or not 0 <= start < end:
            raise ValueError("Invalid seizure onset/offset")
        matching = [r for r in records if (r["subject_id"], r["session_id"]) == (e["subject_id"], e["session_id"])]
        if not any(r["start_sec"] <= start < r["end_sec"] for r in matching):
            raise ValueError("Seizure onset outside recorded intervals; verify event/session alignment")
        result.append(dict(subject_id=e["subject_id"], session_id=e["session_id"], onset_sec=start, offset_sec=end))
    return sorted(result, key=lambda e: (e["subject_id"], e["session_id"], e["onset_sec"]))


def covered_segments(records, record_ids, start, end, split, config, for_history):
    """Require continuous coverage within one split; never bridge even short unknown gaps."""
    # Target interval is (t,t+H], so its RIGHT endpoint must also be observed.
    if not for_history and not any(records[i]["start_sec"] <= end < records[i]["end_sec"]
                                   and records[i]["split"] == split and records[i]["annotation_complete"] for i in record_ids):
        return None
    cursor, segments = start, []
    for record_index in record_ids:
        record = records[record_index]
        left = record["start_sec"] + (config.filter_warmup_seconds if for_history else 0)
        right = record["end_sec"]
        if right <= cursor + 1e-6:
            continue
        if left > cursor + 1e-6:
            return None
        if record["split"] != split or (not for_history and not record["annotation_complete"]):
            return None
        stop = min(right, end)
        if for_history:
            s = (cursor - record["start_sec"]) * config.sample_rate
            e = (stop - record["start_sec"]) * config.sample_rate
            if abs(s - round(s)) > 1e-4 or abs(e - round(e)) > 1e-4:
                raise ValueError("Endpoint is not on the recorded sample grid")
            segments.append([record_index, round(s), round(e)])
        cursor = stop
        if cursor >= end - 1e-6:
            return segments
    return None


def build_index(records, events, config: ForecastConfig):
    groups = {}
    for i, r in enumerate(records):
        groups.setdefault((r["subject_id"], r["session_id"]), []).append(i)
    rows, counts = [], Counter()
    labeled_events = {split: set() for split in ("train", "validation", "test")}
    for event in events:
        record = next(r for r in records if (r["subject_id"], r["session_id"]) == (event["subject_id"], event["session_id"])
                      and r["start_sec"] <= event["onset_sec"] < r["end_sec"])
        counts[f'{record["split"]}_catalog_events'] += 1
    for identity, ids in groups.items():
        group_events = [e for e in events if (e["subject_id"], e["session_id"]) == identity]
        start, end = records[ids[0]]["start_sec"], records[ids[-1]]["end_sec"]
        stride = config.endpoint_stride_seconds
        for t in np.arange(math.ceil(start / stride) * stride, end + 1e-6, stride):
            counts["candidate_endpoints"] += 1
            anchor = next((records[i] for i in reversed(ids) if records[i]["start_sec"] < t <= records[i]["end_sec"] + 1e-6), None)
            if anchor is None:
                counts["no_anchor"] += 1
                continue
            left, right = t - config.history_minutes * 60, t + config.horizon_minutes * 60
            segments = covered_segments(records, ids, left, t, anchor["split"], config, True)
            if segments is None:
                counts["incomplete_history_or_split_boundary"] += 1
                continue
            if any((e["onset_sec"] <= t < e["offset_sec"] + config.postictal_seconds)
                   or (e["onset_sec"] < t and e["offset_sec"] > left) for e in group_events):
                counts["ictal_or_postictal_input"] += 1
                continue
            observed = covered_segments(records, ids, t, right, anchor["split"], config, False) is not None
            target = int(any(t < e["onset_sec"] <= right for e in group_events)) if observed else -1
            if target == 1:
                labeled_events[anchor["split"]].update((*identity, e["onset_sec"], e["offset_sec"])
                                                      for e in group_events if t < e["onset_sec"] <= right)
            rows.append(dict(subject_id=identity[0], session_id=identity[1], record_id=anchor["record_id"],
                             split=anchor["split"], endpoint_sec=float(t), history_start_sec=float(left),
                             forecast_end_sec=float(right), segments=segments, target=target, target_observed=observed))
            counts[f'{anchor["split"]}_target_{target}'] += 1
    counts["input_eligible_endpoints"] = len(rows)
    for split, unique_events in labeled_events.items():
        counts[f"{split}_unique_events_with_positive_endpoints"] = len(unique_events)
    return {"schema_version": 1, "protocol": config.protocol(), "records": records, "events": events,
            "rows": rows, "audit": dict(counts), "score_semantics": "future_event_not_legacy_state"}


def prepare(records_path, events_path, output, config):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing index: {output}")
    records = read_records(records_path, config)
    index = build_index(records, read_events(events_path, records), config)
    index["provenance"] = {"records_csv_sha256": file_sha256(records_path), "events_csv_sha256": file_sha256(events_path)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", required=True, help="Verified record catalog CSV, not old window labels")
    parser.add_argument("--events", required=True, help="Seizure onset/offset CSV in the same session time base")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    index = prepare(args.records, args.events, args.output, load_config(args.config))
    print(json.dumps(index["audit"], indent=2))
    if not any(r["target_observed"] for r in index["rows"]):
        print("WARNING: no fully observed labels. Reconsider the protocol; do NOT turn unknowns into negatives.")


if __name__ == "__main__":
    main()
