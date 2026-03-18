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
  --split SPLIT             HuggingFace split (default: "test")
  --output-dir DIR          Output directory (default: ~/.opendetect)
  --batch-size N            Batch size (default: 128)
  --debug                   Run on first 100 examples only
  --evaluate                Evaluate metrics from the saved scores file instead of running detectors
  --version                 Show version
```

## License

MIT