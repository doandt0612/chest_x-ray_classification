"""Stage 7b: describe where a registered model errs on test. Descriptive only.

Nothing here feeds back into the model or the threshold (D5, D6): the threshold stays the
one chosen on val, and test is only looked at to explain the errors.

Inputs, from cxr.evaluation.evaluate:
  reports/evaluation/<version>/test_predictions.csv and val_predictions.csv

Comparisons (NORMAL images only, since the errors to explain are false positives):
  - prob bands of the false positives: just above the threshold, or confidently wrong?
  - image stats of false positives vs true negatives on test;
  - image stats and prob of test NORMAL vs val NORMAL (a shift between the two sources).
  For each image stat, "auc" = P(a > b) + 0.5 * P(a == b) for a random image from each
  group: 0.5 = no separation, 1.0 or 0.0 = complete separation.
Original size and the label: per split, how well original height and aspect ratio alone
predict the label (from the validation table, no image reading); on test NORMAL, the rank
correlation of prob with height and with aspect ratio, and the FP rate by height third.
These are associations; they cannot show why the model errs (no age data, no intervention).

Writes reports/evaluation/<version>/error_analysis.{json,md} and
reports/figures/<version>_{roc,confusion,prob_normal}.png.

Usage: uv run python -m cxr.evaluation.error_analysis --version v1.0.0
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no display needed; must precede pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

from cxr.monitoring.image_stats import STAT_NAMES, image_stats
from cxr.utils.config import load_config

PROB_BANDS = (0.5, 0.9)  # FP bands: [threshold, 0.5), [0.5, 0.9), [0.9, 1]


def outcomes(label: pd.Series, prob: pd.Series, threshold: float) -> pd.Series:
    """TN / FP / FN / TP per image; positive when prob >= threshold."""
    pred = prob >= threshold
    names = np.select(
        [~pred & (label == 0), pred & (label == 0), ~pred & (label == 1)], ["TN", "FP", "FN"], "TP"
    )
    return pd.Series(names, index=label.index)


def prob_bands(prob: pd.Series, threshold: float) -> dict:
    """Counts of probs in [threshold, 0.5), [0.5, 0.9), [0.9, 1]; probs below threshold are ignored."""
    if threshold >= PROB_BANDS[0]:
        raise ValueError(f"prob_bands needs threshold < {PROB_BANDS[0]}, got {threshold}")
    edges = [threshold, *PROB_BANDS, np.inf]
    labels = [
        f"[{lo:.4g}, {hi:.4g})" if hi != np.inf else f"[{lo:.4g}, 1]" for lo, hi in itertools.pairwise(edges)
    ]
    counts = pd.cut(prob, edges, right=False, labels=labels).value_counts(sort=False)
    return {k: int(v) for k, v in counts.items()}


def compare_stats(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """{stat: {"median_a", "median_b", "auc"}}; auc = P(a > b) + 0.5 * P(a == b) for random a, b."""
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    return {
        name: {
            "median_a": float(a[name].median()),
            "median_b": float(b[name].median()),
            "auc": float(roc_auc_score(y, np.r_[a[name], b[name]])),
        }
        for name in STAT_NAMES
    }


def size_label_auc(sizes: pd.DataFrame) -> dict:
    """{split: {"height", "aspect_ratio"}}: AUC of each size measure alone for predicting PNEUMONIA."""
    sizes = sizes.assign(aspect_ratio=sizes["width"] / sizes["height"])
    return {
        split: {name: float(roc_auc_score(g["label"], g[name])) for name in ("height", "aspect_ratio")}
        for split, g in sizes.groupby("split")
    }


def spearman(x: pd.Series, y: pd.Series) -> float:
    return float(x.rank().corr(y.rank()))


def by_height_third(normal: pd.DataFrame, stats: pd.DataFrame) -> list[dict]:
    """Test NORMAL split into thirds by original height: FP rate and rho(prob, aspect) in each."""
    third = pd.qcut(stats["height"], 3, labels=["small", "middle", "large"])
    rows = []
    for name, idx in stats.groupby(third, observed=True).groups.items():
        rows.append(
            {
                "third": name,
                "n": len(idx),
                "height_min": float(stats.loc[idx, "height"].min()),
                "height_max": float(stats.loc[idx, "height"].max()),
                "fp_rate": float((normal.loc[idx, "outcome"] == "FP").mean()),
                "prob_aspect_spearman": spearman(normal.loc[idx, "prob"], stats.loc[idx, "aspect_ratio"]),
            }
        )
    return rows


def stats_for(paths: pd.Series, image_cfg: dict) -> pd.DataFrame:
    rows = []
    for path in paths:
        with Image.open(path) as img:
            rows.append(image_stats(img, image_cfg))
    return pd.DataFrame(rows, index=paths.index)


def plot_roc(val: pd.DataFrame, test: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for name, df in (("val", val), ("test", test)):
        fpr, tpr, _ = roc_curve(df["label"], df["prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(df['label'], df['prob']):.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--")
    ax.set(xlabel="1 - specificity", ylabel="sensitivity", title="ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(counts: dict, threshold: float, path: Path) -> None:
    matrix = np.array([[counts["TN"], counts["FP"]], [counts["FN"], counts["TP"]]])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > matrix.max() / 2 else "black")
    ticks = ["NORMAL", "PNEUMONIA"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=ticks, yticklabels=ticks)
    ax.set(xlabel="predicted", ylabel="true", title=f"Test, threshold {threshold:.4f}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_prob_normal(val_p: pd.Series, test_p: pd.Series, threshold: float, path: Path) -> None:
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(val_p, bins, alpha=0.6, label=f"val NORMAL (n={len(val_p)})")
    ax.hist(test_p, bins, alpha=0.6, label=f"test NORMAL (n={len(test_p)})")
    ax.axvline(threshold, color="black", lw=1, ls="--", label=f"threshold {threshold:.4f}")
    ax.set_yscale("log")  # the bin at 0 holds most images; log keeps the tail visible
    ax.set(xlabel="P(PNEUMONIA)", ylabel="images (log scale)", title="Predicted probability on NORMAL images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_markdown(r: dict) -> str:
    c, rho = r["test_outcomes"], r["normal_prob_spearman"]
    lines = [
        f"# Error analysis: {r['version']} on test",
        "",
        f"Threshold {r['threshold']:.4f} (chosen on val, unchanged). Descriptive only: nothing here",
        "changes the model or the threshold.",
        "",
        f"- Test outcomes: TN {c['TN']}, FP {c['FP']}, FN {c['FN']}, TP {c['TP']}. Test is kept as",
        "  published, so 6 exact-duplicate NORMAL pairs are counted twice (D2).",
        (
            f"- NORMAL above threshold: val {r['normal_above_threshold']['val']:.2%}, "
            f"test {r['normal_above_threshold']['test']:.2%}"
        ),
        (
            f"- Median prob of NORMAL images: val {r['normal_median_prob']['val']:.4f}, "
            f"test {r['normal_median_prob']['test']:.4f}"
        ),
        (
            f"- The {c['FP']} false positives come from {r['fp_patients']} distinct patients: "
            "they are not clustered by patient"
        ),
        "",
        "## Original image size and the label",
        "",
        "Associations only. Size may go with the children's age or how the film was taken; the data",
        "has no age field, so the cause cannot be separated.",
        "",
        "AUC of one original size measure used alone as a score for PNEUMONIA (0.5 = no",
        "information, below 0.5 = smaller value goes with PNEUMONIA). Train and val are filtered",
        "(D3: too small, extreme aspect, RGB removed); test is not.",
        "",
        "| Split | height | aspect_ratio |",
        "| :-- | --: | --: |",
        *(f"| {k} | {v['height']:.3f} | {v['aspect_ratio']:.3f} |" for k, v in r["size_label_auc"].items()),
        "",
        f"On test NORMAL images, rank correlation of the predicted prob with original height {rho['height']:.3f},",
        f"with aspect ratio {rho['aspect_ratio']:.3f}. By height third:",
        "",
        "| Third | n | Height (px) | FP rate | Rank corr. prob vs aspect ratio |",
        "| :-- | --: | :-- | --: | --: |",
        *(
            f"| {t['third']} | {t['n']} | {t['height_min']:.0f} to {t['height_max']:.0f} "
            f"| {t['fp_rate']:.1%} | {t['prob_aspect_spearman']:.3f} |"
            for t in r["normal_by_height_third"]
        ),
        "",
        "## False positives by predicted probability",
        "",
        "| Band | Count |",
        "| :-- | --: |",
        *(f"| {band} | {n} |" for band, n in r["fp_prob_bands"].items()),
    ]
    for title, key, a, b in (
        ("Image stats: test false positives vs test true negatives", "fp_vs_tn", "FP", "TN"),
        ("Image stats: test NORMAL vs val NORMAL", "test_vs_val_normal", "test", "val"),
    ):
        lines += [
            "",
            f"## {title}",
            "",
            f"`auc` = P({a} > {b}) + 0.5 P(tie) for a random {a} image and a random {b} image.",
            "",
            f"| Stat | Median {a} | Median {b} | auc |",
            "| :-- | --: | --: | --: |",
        ]
        for name, s in r[key].items():
            lines.append(f"| {name} | {s['median_a']:.4g} | {s['median_b']:.4g} | {s['auc']:.3f} |")
    return "\n".join(lines) + "\n"


def analyze(
    test: pd.DataFrame, val: pd.DataFrame, sizes: pd.DataFrame, threshold: float, image_cfg: dict
) -> dict:
    """test/val: prediction rows joined with splits.csv (image_id, label, prob, path, patient_id).

    sizes: split, label, width, height of the original images (train, val, test).
    """
    test = test.assign(outcome=outcomes(test["label"], test["prob"], threshold))
    test_n, val_n = test[test["label"] == 0], val[val["label"] == 0]
    fp, tn = test_n[test_n["outcome"] == "FP"], test_n[test_n["outcome"] == "TN"]
    test_stats, val_stats = stats_for(test_n["path"], image_cfg), stats_for(val_n["path"], image_cfg)
    return {
        "threshold": threshold,
        "test_outcomes": {k: int((test["outcome"] == k).sum()) for k in ("TN", "FP", "FN", "TP")},
        "normal_above_threshold": {
            "val": float((val_n["prob"] >= threshold).mean()),
            "test": float((test_n["prob"] >= threshold).mean()),
        },
        "normal_median_prob": {"val": float(val_n["prob"].median()), "test": float(test_n["prob"].median())},
        "fp_patients": int(fp["patient_id"].nunique()),
        "tn_patients": int(tn["patient_id"].nunique()),
        "normal_prob_spearman": {
            name: spearman(test_n["prob"], test_stats[name]) for name in ("height", "aspect_ratio")
        },
        "normal_by_height_third": by_height_third(test_n, test_stats),
        "size_label_auc": size_label_auc(sizes),
        "fp_prob_bands": prob_bands(fp["prob"], threshold),
        "fp_vs_tn": compare_stats(test_stats.loc[fp.index], test_stats.loc[tn.index]),
        "test_vs_val_normal": compare_stats(test_stats, val_stats),
        "false_positives": fp[["image_id", "prob"]].sort_values("prob").to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", required=True, help="registered version, e.g. v1.0.0")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    with open(Path(args.models_dir) / args.version / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    eval_dir = Path(args.reports_dir) / "evaluation" / args.version
    splits = pd.read_csv(Path(data_cfg["processed_dir"]) / "splits.csv")
    sizes = splits[["image_id", "split", "label"]].merge(
        pd.read_csv(Path(data_cfg["interim_dir"]) / "validation.csv")[["image_id", "width", "height"]],
        on="image_id",
        validate="one_to_one",
    )
    if len(sizes) != len(splits):
        raise ValueError("validation.csv is missing images listed in splits.csv")
    splits = splits[["image_id", "path", "patient_id"]]

    def load(split: str) -> pd.DataFrame:
        preds = pd.read_csv(eval_dir / f"{split}_predictions.csv")
        return preds.merge(splits, on="image_id", how="left", validate="one_to_one")

    test, val = load("test"), load("val")
    threshold = metadata["threshold"]
    report = {"version": args.version, **analyze(test, val, sizes, threshold, metadata["image"])}

    with open(eval_dir / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    (eval_dir / "error_analysis.md").write_text(render_markdown(report), encoding="utf-8")

    fig_dir = Path(args.reports_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_roc(val, test, fig_dir / f"{args.version}_roc.png")
    plot_confusion(report["test_outcomes"], threshold, fig_dir / f"{args.version}_confusion.png")
    plot_prob_normal(
        val.loc[val["label"] == 0, "prob"],
        test.loc[test["label"] == 0, "prob"],
        threshold,
        fig_dir / f"{args.version}_prob_normal.png",
    )
    print(render_markdown(report))
    print(f"Wrote {eval_dir}/error_analysis.{{json,md}} and figures in {fig_dir}")


if __name__ == "__main__":
    main()
