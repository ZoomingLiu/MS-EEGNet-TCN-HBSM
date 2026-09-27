"""Dependency-free SVG diagnostics for merged subject-level HBSI outputs."""

import html
import pickle
import argparse
from pathlib import Path

import numpy as np

try:
    from .risk_calibration_loso import CausalRiskCalibrator
except ImportError:  # pragma: no cover
    from risk_calibration_loso import CausalRiskCalibrator


def find_data_root(start_dir: Path) -> Path:
    candidates = (
        start_dir / "data" / "external" / "member2",
        start_dir / "data",
        start_dir / "data" / "external" / "member3_output128_causal_full_timeline" / "data",
        start_dir.parent / "data",
        start_dir.parent / "member3_output128_causal_full_timeline" / "data",
    )
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("subject_*/labels.npy")):
            return candidate
    raise FileNotFoundError("Cannot find subject directories containing labels.npy.")


def _points(x, y, left, top, width, height):
    if len(x) == 0:
        return ""
    x_min, x_max = float(x[0]), float(x[-1])
    span = max(x_max - x_min, 1e-12)
    px = left + (np.asarray(x) - x_min) / span * width
    py = top + (1.0 - np.clip(np.asarray(y), 0.0, 1.0)) * height
    return " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py))


def _downsample_indices(length, maximum_points=1600):
    if length <= maximum_points:
        return np.arange(length)
    return np.linspace(0, length - 1, maximum_points, dtype=int)


def render_record_svg(output, subject_dir: Path, record_id: str, destination: Path):
    records = np.load(subject_dir / "sequence_records.npy", mmap_mode="r")
    indices = np.flatnonzero(records == record_id)
    if indices.size == 0:
        raise ValueError(f"record {record_id!r} is absent from {subject_dir.name}.")
    labels = np.load(subject_dir / "labels.npy", mmap_mode="r")[indices]
    raw_risk = np.load(subject_dir / "tcn_risk.npy", mmap_mode="r")[indices]
    times = output.timestamps_sec[indices]
    probabilities = output.hsmm_state_probabilities[indices]
    states = output.hsmm_state_path[indices]
    change = output.bocpd_change_probability[indices]
    evidence, _ = CausalRiskCalibrator().transform(raw_risk)
    take = _downsample_indices(len(indices))

    width, height = 1280, 820
    left, plot_width, panel_height = 82, 1140, 185
    tops = (100, 340, 580)
    title = f"Strictly causal HBSI | {output.patient_id} | {record_id} | {len(indices)} endpoints"
    fragments = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="38" text-anchor="middle" font-family="Arial" font-size="20" fill="#111827">{html.escape(title)}</text>',
    ]
    for top in tops:
        fragments.append(f'<rect x="{left}" y="{top}" width="{plot_width}" height="{panel_height}" fill="#ffffff" stroke="#9ca3af"/>')
        for value in (0.0, 0.5, 1.0):
            y = top + (1.0 - value) * panel_height
            fragments.append(f'<line x1="{left}" x2="{left+plot_width}" y1="{y}" y2="{y}" stroke="#e5e7eb"/>')
            fragments.append(f'<text x="{left-12}" y="{y+4}" text-anchor="end" font-family="Arial" font-size="12">{value:g}</text>')

    series = (
        (raw_risk[take], "#6b7280", "TCN risk"),
        (evidence[take], "#2563eb", "causal evidence"),
        (probabilities[take, 2], "#dc2626", "HSMM preictal"),
    )
    for values, color, _ in series:
        fragments.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{_points(times[take], values, left, tops[0], plot_width, panel_height)}"/>')

    normalized_states = states[take] / 2.0
    fragments.append(f'<polyline fill="none" stroke="#7c3aed" stroke-width="1.7" points="{_points(times[take], normalized_states, left, tops[1], plot_width, panel_height)}"/>')
    known = labels[take] >= 0
    if np.any(known):
        label_points = _points(times[take][known], labels[take][known], left, tops[1], plot_width, panel_height)
        fragments.append(f'<polyline fill="none" stroke="#111827" stroke-width="1.2" stroke-dasharray="5 4" points="{label_points}"/>')
    fragments.append(f'<polyline fill="none" stroke="#059669" stroke-width="1.5" points="{_points(times[take], change[take], left, tops[2], plot_width, panel_height)}"/>')

    labels_text = (
        (tops[0] - 18, "TCN risk (gray) · causal evidence (blue) · HSMM preictal probability (red)"),
        (tops[1] - 18, "HSMM path / 2 (purple) · supervised label where available (black dashed)"),
        (tops[2] - 18, "BOCPD probability of a change within the latest 3 endpoints (green)"),
    )
    for y, text_value in labels_text:
        fragments.append(f'<text x="{left}" y="{y}" font-family="Arial" font-size="14" fill="#111827">{text_value}</text>')
    fragments.append(f'<text x="{left+plot_width/2}" y="800" text-anchor="middle" font-family="Arial" font-size="14">Endpoint time (seconds)</text>')
    fragments.append('</svg>')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(fragments), encoding="utf-8")
    return destination


def generate_diagnostic_figures(pkl_path, data_root_dir, output_dir, subjects=None):
    with Path(pkl_path).open("rb") as file_handle:
        outputs = pickle.load(file_handle)
    by_subject = {output.patient_id: output for output in outputs}
    selected = subjects or list(by_subject)
    generated = []
    for subject in selected:
        output = by_subject[subject]
        subject_dir = Path(data_root_dir) / subject
        records = np.load(subject_dir / "sequence_records.npy", mmap_mode="r")
        labels = np.load(subject_dir / "labels.npy", mmap_mode="r")
        unique_records = np.unique(records)
        # Prefer a record containing supervised positives; otherwise use the longest.
        chosen = max(
            unique_records,
            key=lambda record: (int(np.sum(labels[records == record] == 1)), int(np.sum(records == record))),
        )
        destination = Path(output_dir) / f"{subject}_{chosen}.svg"
        generated.append(render_record_svg(output, subject_dir, str(chosen), destination))
    return generated


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render diagnostic SVGs from a TRUSTED HBSI pickle.")
    parser.add_argument("--input", type=Path, help="Trusted local pickle; never use an untrusted pickle.")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--subjects", nargs="+", help="Default: all subjects in the result.")
    args = parser.parse_args(argv)
    if args.input is None or args.data_dir is None or args.output_dir is None:
        try:
            from eeg_pipeline.paths import project_root
            root = project_root()
        except ImportError:
            root = Path(__file__).resolve().parents[2]
    else:
        root = None
    generated = generate_diagnostic_figures(
        args.input if args.input is not None else root / "outputs" / "hbsi" / "hbsi_loso_output_final.pkl",
        args.data_dir if args.data_dir is not None else find_data_root(root),
        args.output_dir if args.output_dir is not None else root / "reports" / "figures" / "hbsi",
        subjects=args.subjects,
    )
    print("\n".join(str(path) for path in generated))


if __name__ == "__main__":
    main()
