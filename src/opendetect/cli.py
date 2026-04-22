"""CLI entry point for OpenDetect.

Usage
-----
    opendetect <DATASET> [--detector NAME] [--text-field FIELD] ...
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import torch

from opendetect import __version__
from opendetect.config import get_output_dir
from opendetect.data import load_dataset

# Import detectors to trigger registration
import opendetect.detectors  # noqa: F401
from opendetect.registry import get_all_detectors, get_detector, list_detectors

logger = logging.getLogger("opendetect")


def _release_gpu_memory() -> None:
    """Run a full GC pass and empty PyTorch's CUDA caching allocator.

    Call this after a detector finishes so freed GPU blocks return to
    the pool before the next detector tries to load its weights.
    Safe to call even when torch is unavailable or running on CPU.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
        "--reviser",
        type=str,
        choices=["hf", "vllm", "openai"],
        default="hf",
        help=(
            "Reviser backend for revise-detect.  'hf' uses "
            "transformers; 'vllm' serves the same model via vLLM for "
            "much faster batched generation (default model "
            "Qwen2.5-7B-Instruct); 'openai' calls the OpenAI API "
            "(requires OPENAI_API_KEY)."
        ),
    )
    parser.add_argument(
        "--reviser-model",
        type=str,
        default=None,
        help=(
            "Override the reviser model identifier.  Defaults: "
            "Qwen/Qwen2.5-7B-Instruct (hf/vllm), gpt-4o-mini (openai)."
        ),
    )
    parser.add_argument(
        "--dna-gpt-regenerator",
        type=str,
        choices=["hf", "vllm", "openai"],
        default="hf",
        help=(
            "Regenerator backend for dna-gpt.  'hf' uses transformers; "
            "'vllm' serves the same model via vLLM (recommended for "
            "K>1 continuations); 'openai' uses the OpenAI completions "
            "API (default gpt-3.5-turbo-instruct)."
        ),
    )
    parser.add_argument(
        "--dna-gpt-regenerator-model",
        type=str,
        default=None,
        help=(
            "Override the dna-gpt regenerator model identifier. "
            "Defaults: Qwen/Qwen2.5-7B-Instruct (hf/vllm), "
            "gpt-3.5-turbo-instruct (openai)."
        ),
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=4096,
        help=(
            "vLLM max_model_len (cap on prompt + generated tokens) "
            "for revise-detect / dna-gpt when --reviser vllm or "
            "--dna-gpt-regenerator vllm is set.  Default: 4096."
        ),
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.9,
        help=(
            "vLLM gpu_memory_utilization fraction for revise-detect / "
            "dna-gpt when a vllm backend is selected.  Lower if other "
            "processes share the GPU.  Default: 0.9."
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
        from opendetect.metrics import get_tpr_target
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
            if name in df.columns and not args.force:
                logger.info("Skipping %s — already computed.", name)
                continue

            detector_cls = get_detector(name)
            detector = detector_cls()

            logger.info("Running detector: %s", name)
            try:
                kwargs: dict = {"batch_size": args.batch_size}
                if name == "revise-detect":
                    kwargs["reviser"] = args.reviser
                    if args.reviser_model is not None:
                        kwargs["reviser_model"] = args.reviser_model
                    if args.reviser == "vllm":
                        kwargs["vllm_max_model_len"] = args.vllm_max_model_len
                        kwargs["vllm_gpu_memory_utilization"] = (
                            args.vllm_gpu_memory_utilization
                        )
                if name == "dna-gpt":
                    kwargs["regenerator"] = args.dna_gpt_regenerator
                    if args.dna_gpt_regenerator_model is not None:
                        kwargs["regenerator_model"] = (
                            args.dna_gpt_regenerator_model
                        )
                    if args.dna_gpt_regenerator == "vllm":
                        kwargs["vllm_max_model_len"] = args.vllm_max_model_len
                        kwargs["vllm_gpu_memory_utilization"] = (
                            args.vllm_gpu_memory_utilization
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
            else:
                # Save incrementally after each successful detector.
                df.to_json(output_path, orient="records", lines=True)
            finally:
                # Drop the detector's GPU allocation before the next
                # one loads; otherwise full-pipeline runs OOM.
                detector.teardown()
                del detector
                _release_gpu_memory()

        # --- Final save -----------------------------------------------------------
        df.to_json(output_path, orient="records", lines=True)
        logger.info("Results saved to: %s", output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

