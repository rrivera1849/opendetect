"""CLI entry point for OpenDetect.

Usage
-----
    opendetect <DATASET> [--detector NAME] [--text-field FIELD] ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from opendetect import __version__
from opendetect.config import get_output_dir
from opendetect.data import extract_fewshot, load_dataset

# Import detectors to trigger registration
import opendetect.detectors  # noqa: F401
from opendetect.registry import get_all_detectors, get_detector, list_detectors

logger = logging.getLogger("opendetect")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opendetect",
        description="Run machine-text detection algorithms on a dataset.",
    )
    parser.add_argument(
        "dataset",
        help=(
            "Path to a JSONL file, or a HuggingFace dataset identifier "
            "(e.g. 'username/dataset_name')."
        ),
    )
    parser.add_argument(
        "--detector",
        type=str,
        default=None,
        help=(
            "Name of a specific detector to run.  "
            "If omitted, all registered detectors are executed.  "
            f"Available: {', '.join(list_detectors())}"
        ),
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Name of the column containing text to classify (default: 'text').",
    )
    parser.add_argument(
        "--label-field",
        type=str,
        default="label",
        help="Name of the label column (default: 'label').",
    )
    parser.add_argument(
        "--machine-label",
        type=int,
        default=1,
        help="Value indicating machine-generated text (default: 1).",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help=(
            "Number of machine-text examples to extract as few-shot background "
            "for style-based detectors (default: 0)."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="HuggingFace dataset split to load (default: 'test').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: ~/.opendetect or $OPENDETECT_OUTPUT_DIR).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for detectors that support batching (default: 128).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run on first 100 examples only.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate metrics from the saved scores file instead of running detectors.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _dataset_key(source: str) -> str:
    """Derive a filesystem-safe key from the dataset source."""
    return Path(source).stem if Path(source).exists() else source.replace("/", "_")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    # --- Determine output path ------------------------------------------------
    output_dir = Path(args.output_dir) if args.output_dir else get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_key = _dataset_key(args.dataset)
    suffix = ".debug" if args.debug else ""
    output_path = output_dir / f"{ds_key}_scores{suffix}.jsonl"

    if args.evaluate:
        if not output_path.exists():
            logger.error("Saved scores file not found: %s", output_path)
            return 1
            
        logger.info("Evaluating results from %s", output_path)
        df = pd.read_json(output_path, lines=True)
        
        if args.label_field not in df.columns:
            logger.error("Dataset lacks the label column '%s' required for evaluation.", args.label_field)
            return 1

        machine_df = df[df[args.label_field] == args.machine_label]
        human_df = df[df[args.label_field] != args.machine_label]
        
        from opendetect.metrics import get_tpr_target
        from sklearn.metrics import roc_auc_score, roc_curve
        
        results = []
        for det in list_detectors():
            if det in df.columns:
                human_scores = human_df[det].dropna().tolist()
                machine_scores = machine_df[det].dropna().tolist()
                
                if not human_scores or not machine_scores:
                    continue
                
                scores = human_scores + machine_scores
                labels = [0] * len(human_scores) + [1] * len(machine_scores)
                fpr, tpr, _ = roc_curve(labels, scores)
                roc_auc = roc_auc_score(labels, scores)

                row = {"Detector": det.upper(), "ROC AUC": f"{roc_auc:.4f}"}
                
                for target_fpr in [0.001, 0.01, 0.1]:
                    roc_auc_at_fpr = roc_auc_score(labels, scores, max_fpr=target_fpr)
                    row[f"AUROC({target_fpr*100})"] = f"{roc_auc_at_fpr:.4f}"
                    
                for target_fpr in [0.001, 0.01, 0.1]:
                    tpr_val = get_tpr_target(fpr, tpr, target_fpr)
                    row[f"TPR(FPR={target_fpr*100}%)"] = f"{tpr_val:5.1f}%"
                
                results.append(row)
                
        if results:
            results_df = pd.DataFrame(results)
            print("\nEvaluation Results:")
            print("-" * 105)
            print(results_df.to_markdown(index=False))
            print("-" * 105 + "\n")
        return 0


    # --- Load dataset ---------------------------------------------------------
    logger.info("Loading dataset: %s", args.dataset)
    df = load_dataset(
        args.dataset,
        text_field=args.text_field,
        label_field=args.label_field,
        split=args.split,
    )

    if args.debug:
        df = df.head(100)
        logger.info("Debug mode — using first 100 examples.")

    # --- Few-shot extraction --------------------------------------------------
    fewshot_texts: list[str] = []
    if args.num_fewshot > 0:
        df, fewshot_texts = extract_fewshot(
            df,
            num_fewshot=args.num_fewshot,
            label_field=args.label_field,
            machine_label=args.machine_label,
            text_field=args.text_field,
        )

    # Filter out rows with empty text
    df = df[df[args.text_field].apply(lambda t: isinstance(t, str) and len(t.strip()) > 0)]
    df = df.reset_index(drop=True)
    texts = df[args.text_field].tolist()
    logger.info("Scoring %d texts.", len(texts))

    # --- Resume from existing scores -----------------------------------------
    if output_path.exists():
        logger.info("Found existing output at %s — loading.", output_path)
        existing_df = pd.read_json(output_path, lines=True)
        # Merge previously computed detector columns into the dataframe
        for col in existing_df.columns:
            if col not in df.columns:
                df[col] = existing_df[col]

    # --- Determine which detectors to run ------------------------------------
    if args.detector:
        detector_names = [args.detector]
    else:
        detector_names = list_detectors()

    # --- Run detectors --------------------------------------------------------
    for name in detector_names:
        if name in df.columns:
            logger.info("Skipping %s — already computed.", name)
            continue

        detector_cls = get_detector(name)
        detector = detector_cls()

        # Skip fewshot-required detectors if no fewshot available
        if detector.requires_fewshot and not fewshot_texts:
            logger.warning(
                "Skipping %s — requires --num-fewshot > 0.",
                name,
            )
            continue

        logger.info("Running detector: %s", name)
        try:
            kwargs: dict = {"batch_size": args.batch_size}
            if detector.requires_fewshot:
                kwargs["background_texts"] = fewshot_texts
            scores = detector.score(texts, **kwargs)
            df[name] = scores
        except Exception:
            logger.exception("Detector %s failed — skipping.", name)
            continue

        # Save incrementally after each detector
        df.to_json(output_path, orient="records", lines=True)

    # --- Final save -----------------------------------------------------------
    df.to_json(output_path, orient="records", lines=True)
    logger.info("Results saved to: %s", output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

