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
from opendetect.data import load_dataset

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
        nargs="+",
        help=(
            "Path to one or more JSONL files, or HuggingFace dataset identifiers "
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
        "--force",
        action="store_true",
        help="Force re-run of detectors even if results already exist.",
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
        "--num-trials",
        type=int,
        default=100,
        help=(
            "Number of random support-set trials for style-based detectors "
            "(default: 100)."
        ),
    )
    parser.add_argument(
        "--source-field",
        type=str,
        default=None,
        help=(
            "Column name for machine source annotation.  Enables multi-target "
            "mode for style-based detectors."
        ),
    )
    parser.add_argument(
        "--source-preprocess",
        type=str,
        default=None,
        help=(
            "Python lambda expression to preprocess the source field, "
            "e.g. \"lambda x: x.split('/')[-1]\"."
        ),
    )
    parser.add_argument(
        "--multitarget-mode",
        type=str,
        choices=["max", "instance", "single", "domain", "domain-single"],
        default="max",
        help=(
            "How to use --source-field for style detectors.  'max' (default) "
            "computes per-class aggregate embeddings and scores by max cosine "
            "similarity.  'instance' skips aggregation entirely and scores by "
            "max cosine similarity to any individual support embedding (LUAR's "
            "original nearest-instance rule; diverges from 'max' when "
            "--num-fewshot > 1).  'single' uses --source-field only to "
            "stratify the support set, then pools all stratified samples into "
            "one mean centroid.  'domain' restricts scoring to same-domain "
            "centroids — each query is scored against class centroids built "
            "from support samples sharing its domain (requires "
            "--domain-field).  'domain-single' samples per (domain, class) "
            "pair but pools each domain's samples into one mean centroid — "
            "queries score against their own domain's single centroid "
            "(requires --domain-field)."
        ),
    )
    parser.add_argument(
        "--domain-field",
        type=str,
        default=None,
        help=(
            "Column name containing domain information for all texts "
            "(human and machine).  Used with --multitarget-mode domain "
            "to restrict scoring to same-domain centroids."
        ),
    )
    parser.add_argument(
        "--domain-preprocess",
        type=str,
        default=None,
        help=(
            "Python lambda expression to extract the domain from "
            "--domain-field values, e.g. "
            "\"lambda x: x.split('_machine')[0].split('_human')[0]\"."
        ),
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help=(
            "If set, aggregate queries into groups of this size before "
            "scoring.  Groups are homogeneous on human-vs-machine, on "
            "source class (machines, when --source-field is provided), "
            "and on domain (in domain / domain-single modes).  Remainders "
            "smaller than the group size are dropped.  Support samples are "
            "excluded from the group pool per trial.  The NPZ score matrix "
            "has shape (num_trials, num_groups) with a parallel "
            "group_labels vector."
        ),
    )
    parser.add_argument(
        "--dna-gpt-regenerator",
        type=str,
        choices=["hf", "openai"],
        default="hf",
        help=(
            "Regenerator backend for dna-gpt.  'hf' uses a local "
            "HuggingFace chat model (default Qwen2.5-7B-Instruct, via "
            "vLLM if installed); 'openai' uses the OpenAI completions "
            "API (default gpt-3.5-turbo-instruct)."
        ),
    )
    parser.add_argument(
        "--dna-gpt-regenerator-model",
        type=str,
        default=None,
        help=(
            "Override the dna-gpt regenerator model identifier. "
            "Defaults: Qwen/Qwen2.5-7B-Instruct (hf), "
            "gpt-3.5-turbo-instruct (openai)."
        ),
    )
    parser.add_argument(
        "--dna-gpt-k",
        type=int,
        default=20,
        help="Number of regenerations per text for dna-gpt (default: 20).",
    )
    parser.add_argument(
        "--dna-gpt-truncate-ratio",
        type=float,
        default=0.5,
        help=(
            "Fraction of each text (by whitespace-split words) used as "
            "the regeneration prefix for dna-gpt (default: 0.5)."
        ),
    )
    parser.add_argument(
        "--dna-gpt-prompt-field",
        type=str,
        default=None,
        help=(
            "Optional dataset column containing the original prompt for "
            "each text.  When set, dna-gpt prepends the prompt to the "
            "truncated prefix before regeneration."
        ),
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

    if args.evaluate:
        import numpy as np

        from opendetect.metrics import evaluate_score_matrix, get_tpr_target
        from sklearn.metrics import roc_auc_score, roc_curve

        detector_metrics_all = {}

        for dataset in args.dataset:
            ds_key = _dataset_key(dataset)
            suffix = ".debug" if args.debug else ""
            output_path = output_dir / f"{ds_key}_scores{suffix}.jsonl"

            if not output_path.exists():
                logger.error("Saved scores file not found: %s", output_path)
                continue

            logger.info("Evaluating results from %s", output_path)
            df = pd.read_json(output_path, lines=True)

            if args.label_field not in df.columns:
                logger.error(
                    "Dataset lacks the label column '%s' required for evaluation.",
                    args.label_field,
                )
                continue

            machine_df = df[df[args.label_field] == args.machine_label]
            human_df = df[df[args.label_field] != args.machine_label]

            results = []
            for det in list_detectors():
                if det not in df.columns:
                    continue

                det_upper = det.upper()
                if det_upper not in detector_metrics_all:
                    detector_metrics_all[det_upper] = {}

                # Check for NPZ score matrix (styledetect multi-trial)
                npz_path = output_path.with_suffix(f".{det}.npz")
                if npz_path.exists():
                    npz = np.load(npz_path, allow_pickle=True)
                    score_matrix = npz["scores"]
                    # Grouped runs carry a per-column label vector in the NPZ;
                    # otherwise fall back to the per-sample label column.
                    if "group_labels" in npz.files:
                        labels_arr = npz["group_labels"]
                    else:
                        labels_arr = np.array(
                            (df[args.label_field] == args.machine_label).astype(int),
                        )
                    trial_avg = evaluate_score_matrix(
                        score_matrix,
                        labels_arr,
                    )
                    if not trial_avg:
                        continue
                    row = {
                        "Detector": det_upper,
                        "ROC AUC": f"{trial_avg['ROC AUC']:.4f}",
                    }
                    detector_metrics_all[det_upper].setdefault(
                        "ROC AUC", [],
                    ).append(trial_avg["ROC AUC"])
                    for target_fpr in [0.001, 0.01, 0.1]:
                        auroc_key = f"AUROC({target_fpr * 100})"
                        row[auroc_key] = f"{trial_avg[auroc_key]:.4f}"
                        detector_metrics_all[det_upper].setdefault(
                            auroc_key, [],
                        ).append(trial_avg[auroc_key])
                        tpr_key = f"TPR(FPR={target_fpr * 100}%)"
                        row[tpr_key] = f"{trial_avg[tpr_key]:5.1f}%"
                        detector_metrics_all[det_upper].setdefault(
                            tpr_key, [],
                        ).append(trial_avg[tpr_key])
                    results.append(row)
                    continue

                # Standard single-score evaluation
                human_scores = human_df[det].dropna().tolist()
                machine_scores = machine_df[det].dropna().tolist()

                if not human_scores or not machine_scores:
                    continue

                scores = human_scores + machine_scores
                labels = [0] * len(human_scores) + [1] * len(machine_scores)
                fpr, tpr, _ = roc_curve(labels, scores)
                roc_auc = float(roc_auc_score(labels, scores))

                row = {"Detector": det_upper, "ROC AUC": f"{roc_auc:.4f}"}

                detector_metrics_all[det_upper].setdefault("ROC AUC", []).append(roc_auc)

                for target_fpr in [0.001, 0.01, 0.1]:
                    roc_auc_at_fpr = float(roc_auc_score(labels, scores, max_fpr=target_fpr))
                    metric_name = f"AUROC({target_fpr*100})"
                    row[metric_name] = f"{roc_auc_at_fpr:.4f}"
                    detector_metrics_all[det_upper].setdefault(metric_name, []).append(roc_auc_at_fpr)

                for target_fpr in [0.001, 0.01, 0.1]:
                    tpr_val = float(get_tpr_target(fpr, tpr, target_fpr))
                    metric_name = f"TPR(FPR={target_fpr*100}%)"
                    row[metric_name] = f"{tpr_val:5.1f}%"
                    detector_metrics_all[det_upper].setdefault(metric_name, []).append(tpr_val)

                results.append(row)

            if results:
                results_df = pd.DataFrame(results)
                print(f"\nEvaluation Results for {dataset}:")
                print("-" * 105)
                print(results_df.to_markdown(index=False))
                print("-" * 105 + "\n")
                
        if len(args.dataset) > 1 and detector_metrics_all:
            macro_results = []
            for det in list_detectors():
                det_upper = det.upper()
                if det_upper not in detector_metrics_all:
                    continue
                metrics = detector_metrics_all[det_upper]
                row = {"Detector": det_upper}
                avg_roc = sum(metrics["ROC AUC"]) / len(metrics["ROC AUC"])
                row["ROC AUC"] = f"{avg_roc:.4f}"
                for target_fpr in [0.001, 0.01, 0.1]:
                    metric_name = f"AUROC({target_fpr*100})"
                    avg_val = sum(metrics[metric_name]) / len(metrics[metric_name])
                    row[metric_name] = f"{avg_val:.4f}"
                for target_fpr in [0.001, 0.01, 0.1]:
                    metric_name = f"TPR(FPR={target_fpr*100}%)"
                    avg_val = sum(metrics[metric_name]) / len(metrics[metric_name])
                    row[metric_name] = f"{avg_val:5.1f}%"
                macro_results.append(row)
                
            if macro_results:
                macro_results_df = pd.DataFrame(macro_results)
                print("\nMacro Average Evaluation Results:")
                print("-" * 105)
                print(macro_results_df.to_markdown(index=False))
                print("-" * 105 + "\n")
                
        return 0

    # --- Loop over datasets for running detectors -----------------------------
    for dataset in args.dataset:
        ds_key = _dataset_key(dataset)
        suffix = ".debug" if args.debug else ""
        output_path = output_dir / f"{ds_key}_scores{suffix}.jsonl"

        # --- Load dataset ---------------------------------------------------------
        logger.info("Loading dataset: %s", dataset)
        df = load_dataset(
            dataset,
            text_field=args.text_field,
            label_field=args.label_field,
            split=args.split,
        )

        if args.debug:
            df = df.head(100)
            logger.info("Debug mode — using first 100 examples.")

        # --- Few-shot -------------------------------------------------------------
        # All current few-shot detectors manage their own support-set sampling,
        # so we no longer extract and remove fixed few-shot rows here.

        # Filter out rows with empty text
        df = df[df[args.text_field].apply(lambda t: isinstance(t, str) and len(t.strip()) > 0)]
        df = df.reset_index(drop=True)
        texts = df[args.text_field].tolist()
        logger.info("Scoring %d texts.", len(texts))

        # --- Resume from existing scores -----------------------------------------
        if output_path.exists():
            logger.info("Found existing output at %s — loading.", output_path)
            existing_df = pd.read_json(output_path, lines=True)
            # TODO: Migrate all detector score saving to NPZ format.
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
            if name in df.columns and not args.force:
                logger.info("Skipping %s — already computed.", name)
                continue

            detector_cls = get_detector(name)
            detector = detector_cls()

            # Skip fewshot-required detectors if no fewshot available
            if detector.requires_fewshot and args.num_fewshot <= 0:
                logger.warning(
                    "Skipping %s — requires --num-fewshot > 0.",
                    name,
                )
                continue

            logger.info("Running detector: %s", name)
            try:
                kwargs: dict = {"batch_size": args.batch_size}
                if detector.requires_fewshot:
                    machine_mask = df[args.label_field] == args.machine_label
                    kwargs["machine_indices"] = df.index[machine_mask].tolist()
                    kwargs["num_trials"] = args.num_trials
                    kwargs["num_fewshot"] = args.num_fewshot
                    kwargs["output_path"] = output_path.with_suffix(
                        f".{name}.npz",
                    )
                    if args.source_field:
                        source_vals = df.loc[
                            machine_mask, args.source_field
                        ].tolist()
                        if args.source_preprocess:
                            fn = eval(args.source_preprocess)  # noqa: S307
                            source_vals = [fn(x) for x in source_vals]
                        kwargs["source_labels"] = source_vals
                        kwargs["multitarget_mode"] = args.multitarget_mode
                    if args.domain_field:
                        domain_vals = df[args.domain_field].tolist()
                        if args.domain_preprocess:
                            fn = eval(args.domain_preprocess)  # noqa: S307
                            domain_vals = [fn(x) for x in domain_vals]
                        kwargs["domain_labels"] = domain_vals
                    if args.group_size is not None:
                        kwargs["group_size"] = args.group_size
                if name == "dna-gpt":
                    kwargs["regenerator"] = args.dna_gpt_regenerator
                    if args.dna_gpt_regenerator_model is not None:
                        kwargs["regenerator_model"] = (
                            args.dna_gpt_regenerator_model
                        )
                    kwargs["K"] = args.dna_gpt_k
                    kwargs["truncate_ratio"] = args.dna_gpt_truncate_ratio
                    if args.dna_gpt_prompt_field:
                        if args.dna_gpt_prompt_field not in df.columns:
                            logger.warning(
                                "dna-gpt: prompt field %r not in dataset; "
                                "running without prompts.",
                                args.dna_gpt_prompt_field,
                            )
                        else:
                            kwargs["prompts"] = (
                                df[args.dna_gpt_prompt_field]
                                .fillna("")
                                .astype(str)
                                .tolist()
                            )
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

