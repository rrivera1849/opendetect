# 🔍 OpenDetect

A modular, extensible toolkit for machine-text detection. Run multiple detection algorithms on any dataset with a single command and evaluate their performance effortlessly!

## Installation

```bash
# Clone the repository
git clone https://github.com/rrivera1849/opendetect.git
cd opendetect

# Install dependencies and CLI package
pip install -e .
```

## Quick Start

OpenDetect natively supports HuggingFace datasets. For example, to run detection on the `yaful/MAGE` dataset:

### Run all detectors

```bash
opendetect yaful/MAGE --split test
```

### Run a single detector

You can choose instead to run a specific detector, for example the `radar` detector:

```bash
opendetect yaful/MAGE --detector radar --split test
```

### Evaluate performance

Once you have generated the detector scores for a dataset, you can evaluate the TPR and ROC AUC metrics seamlessly:

```bash
opendetect yaful/MAGE --evaluate
```

*(This reads the cached results from your previous runs and outputs a metric table without having to load the detectors again).*

### Custom output directory

Results are saved to `~/.opendetect/` by default. You can change this behavior:

```bash
# Via flag
opendetect yaful/MAGE --output-dir ./results

# Or via environment variable
export OPENDETECT_OUTPUT_DIR=./results
opendetect yaful/MAGE
```

## Using Custom JSONL Files

If you have local data instead of a HuggingFace dataset, you can simply pass the path to a `.jsonl` file.

```bash
opendetect my_local_data.jsonl --detector lrr
```

### Dataset Format

OpenDetect expects the local JSONL datasets to contain at least a **text** column. For evaluation, a **label** column is also needed (e.g., `0` for human, `1` for machine). 

```json
{"text": "This is a sample human text.", "label": 0}
{"text": "This was written by an AI model.", "label": 1}
```

If your local or HuggingFace dataset uses a different label-naming convention (like `is_machine`), you can map it instantly:

```bash
opendetect my_local_data.jsonl --label-field is_machine --machine-label 1 --evaluate
```

## Available Detectors

| Detector | Name | Type | Description |
|---|---|---|---|
| Binoculars | `binoculars` | Zero-shot | Observer/performer cross-perplexity ratio |
| FastDetectGPT | `fast-detect-gpt` | Zero-shot | Sampling discrepancy analytic |
| RADAR | `radar` | Supervised | `TrustSafeAI/RADAR-Vicuna-7B` classifier |
| ReMoDetect | `remodetect` | Supervised | `hyunseoki/ReMoDetect-deberta` classifier |
| Rank | `rank` | Zero-shot | Average token rank (GPT-2 XL) |
| LogRank | `log-rank` | Zero-shot | Average log-token rank (GPT-2 XL) |
| LRR | `lrr` | Zero-shot | Log-Likelihood to Log-Rank Ratio (GPT-2 XL) |
| MAGE | `mage` | Supervised | `yaful/MAGE` Longformer classifier |
| NPR | `npr` | Zero-shot | Normalized Perturbation Rank (GPT-2 XL + T5) |
| StyleDetect (CISR) | `styledetect-cisr` | Few-shot | Cosine similarity to a support-set centroid using `AnnaWegmann/Style-Embedding` |
| StyleDetect (SD) | `styledetect-sd` | Few-shot | Same as above, with `StyleDistance/styledistance` |
| SemDetect | `semdetect` | Few-shot | Same as above, with `all-mpnet-base-v2` (semantic) |

## Few-Shot Detection

Few-shot detectors score each text by its cosine similarity to a **support set** of known machine texts embedded into a style/semantic space. Because a single random support set is noisy, OpenDetect samples `--num-trials` independent support sets and saves the full score matrix of shape `(num_trials, num_samples)` as an `.npz` alongside the usual JSONL (the JSONL column stores the per-text mean across trials). Evaluation averages metrics *per trial* before reporting.

### Basic use

```bash
opendetect my_local_data.jsonl \
    --detector styledetect-cisr \
    --num-fewshot 10 \
    --num-trials 100
```

`--num-fewshot` is the support-set size **per trial** (or per class / per `(domain, class)` pair when stratified — see below). Self-exclusion is automatic: samples in a trial's support set are NaN-masked in that trial's scores.

### Multi-target mode

When machine texts come from several sources (e.g. different LLMs) you can pass `--source-field` to stratify the support set per class. `--source-preprocess` is an optional lambda that normalizes the raw field value into a class label.

```bash
opendetect data/Deepfake/cross_domains_cross_models/test.csv \
    --text-field text --label-field label --machine-label 0 \
    --source-field src \
    --source-preprocess "lambda x: x.split('_machine_', 1)[1].split('_', 1)[1]" \
    --multitarget-mode max \
    --num-fewshot 10 --num-trials 100 \
    --detector styledetect-sd
```

`--multitarget-mode` controls how the stratified support set is used at scoring time:

| Mode | Sampling | Scoring |
|---|---|---|
| `max` (default) | `K` per class | max cosine similarity across per-class centroids |
| `single` | `K` per class | pool all stratified samples into one mean centroid |
| `domain` | `K` per `(domain, class)` pair | max similarity across same-domain class centroids (requires `--domain-field`) |
| `domain-single` | `K` per `(domain, class)` pair | similarity to a single same-domain centroid pooled across classes (requires `--domain-field`) |

The `domain` and `domain-single` modes eliminate cross-domain confounds when your style backbone is domain-sensitive. Supply the domain column via `--domain-field` and (optionally) `--domain-preprocess`:

```bash
opendetect data/Deepfake/cross_domains_cross_models/test.csv \
    --text-field text --label-field label --machine-label 0 \
    --source-field src \
    --source-preprocess "lambda x: x.split('_machine_', 1)[1].split('_', 1)[1]" \
    --domain-field src \
    --domain-preprocess "lambda x: x.split('_machine')[0].split('_human')[0]" \
    --multitarget-mode domain \
    --num-fewshot 10 --num-trials 100 \
    --detector styledetect-sd
```

See `examples/run_styledetect_deepfake_domain_multitarget*.sh` for ready-to-run scripts.

### Evaluation

`opendetect <dataset> --evaluate` automatically detects the `.npz` score matrix and reports per-trial ROC AUC, AUROC@FPR, and TPR@FPR averaged across trials.

## Python API

You can also use OpenDetect programmatically:

```python
from opendetect.data import load_dataset
from opendetect.registry import get_detector, list_detectors

# See all available detectors
print(list_detectors())

# Load data (handles HF auto-download or local files!)
df = load_dataset("yaful/MAGE")
texts = df["text"].tolist()

# Run a specific detector
RadarCls = get_detector("radar")
detector = RadarCls()
scores = detector.score(texts)
```

## Output

Runs are **incremental** — if a results file already exists, previously computed detector scores are preserved and only missing detectors are executed.

Results are saved as JSONL with the original text and one column per detector:

```json
{"text": "This is a sample.", "radar": 0.92, "rank": 42.1, "log-rank": 3.21}
```

## Adding a New Detector

Create a new file in `src/opendetect/detectors/`:

```python
# src/opendetect/detectors/my_detector.py
import torch
from opendetect.detectors.base import BaseDetector
from opendetect.registry import register_detector

@register_detector("my-detector")
class MyDetector(BaseDetector):
    def score(self, texts: list[str], **kwargs) -> list[float]:
        # Your detection logic here
        return [0.0] * len(texts)
```

The detector is automatically discovered and registered — no other code changes needed. It will appear in `opendetect --help` and can be run via `opendetect yaful/MAGE --detector my-detector`.

## CLI Reference

```text
opendetect <DATASET> [OPTIONS]

Positional:
  DATASET                   Path to JSONL file or HuggingFace dataset name

Options:
  --detector NAME           Run a specific detector (default: run all)
  --text-field FIELD        Text column name (default: "text")
  --label-field FIELD       Label column name (default: "label")
  --machine-label VALUE     Label value for machine text (default: 1)
  --num-fewshot N           Few-shot examples for style detectors (default: 0)
  --num-trials N            Random support-set trials for few-shot detectors (default: 100)
  --source-field FIELD      Machine-source column for per-class stratification
  --source-preprocess EXPR  Python lambda to normalize --source-field values
  --multitarget-mode MODE   max | single | domain | domain-single  (default: max)
  --domain-field FIELD      Domain column (all texts); enables domain-restricted modes
  --domain-preprocess EXPR  Python lambda to extract the domain from --domain-field
  --split SPLIT             HuggingFace split (default: "test")
  --output-dir DIR          Output directory (default: ~/.opendetect)
  --batch-size N            Batch size (default: 128)
  --debug                   Run on first 100 examples only
  --evaluate                Evaluate metrics from the saved scores file instead of running detectors
  --version                 Show version
```

## License

MIT