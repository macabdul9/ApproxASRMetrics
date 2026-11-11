# On the Robust Approximation of ASR Metrics

Code for our Findings of ACL 2025 paper [**On the Robust Approximation of ASR Metrics**](https://aclanthology.org/2025.findings-acl.1187/).

**Abdul Waheed<sup>1</sup>, Hanin Atwany<sup>2</sup>, Rita Singh<sup>1</sup>, and Bhiksha Raj<sup>1</sup>**

<sup>1</sup>Carnegie Mellon University, <sup>2</sup>MBZUAI

[![Paper](https://img.shields.io/badge/Paper-ACL%20Findings%202025-blue?logo=readthedocs&logoColor=white)](https://aclanthology.org/2025.findings-acl.1187/)
[![arXiv](https://img.shields.io/badge/arXiv-2502.12408-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2502.12408)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white)](https://github.com/macabdul9/ApproxASRMetrics)

We estimate WER and CER without reference transcripts. Each utterance gets two
features: the cosine similarity between SONAR embeddings of the audio and the ASR
hypothesis, and the error rate of the hypothesis measured against a proxy reference,
which is the transcript from the best other ASR model. A stacked regressor maps these
features to error counts, which we normalize by reference length to get approximate
WER/CER (aWER/aCER). The paper evaluates 41 ASR models on 10 standard benchmarks and
on in-the-wild test sets.

## Installation

Use Python 3.10 or newer. Run all commands from the repository root. Stages need
different environments.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The core package (scoring, proxy features, regression, result tables) needs only
NumPy, pandas, scikit-learn, jiwer, and datasets. It runs on CPU.

ASR inference and the eWER3 baseline need PyTorch, Transformers, NeMo, and SpeechBrain:

```bash
python -m pip install -r requirements/asr.txt
```

SONAR embeddings (the similarity feature and the CMD analysis) need `sonar-space` and
`fairseq2`, which pin their own PyTorch versions. Install them in a separate environment:

```bash
python -m pip install -e .
python -m pip install -r requirements/sonar.txt
```

The original experiments did not record package versions, so the requirement files
are unpinned. GPU stages need a smoke test on your CUDA stack.

For figures, install the plotting extra: `python -m pip install -e '.[plot]'`.

## Data

Every evaluation set is a 1,000-utterance subset of a public corpus. The paper used
these processed subsets, whose Hub IDs appear in `configs/datasets_*.json`
(`macabdul9/<name>_eval_1K`). They may not be publicly accessible. Several sources
(Common Voice 17, GigaSpeech, SPGISpeech, People's Speech) are gated and may not allow
redistribution. The primary path is to rebuild the subsets yourself:

```bash
bash scripts/build_subsets.sh \
    --source-config configs/source_datasets.json \
    --output-dir data/subsets
```

Each subset keeps utterances of 1 to 30 seconds whose normalized reference is
non-empty. It then shuffles twice with seed 42, takes the first 1,000, resamples to
16 kHz, and keeps the `audio`, `reference`, and `id` columns. Add
`--push-to-hub NAMESPACE` to also upload each subset as a private dataset.

`configs/source_datasets.json` maps a source dataset ID to its `config`, `split`, and
`text_column`. Only Common Voice 17 has a recorded Hub ID. The original code stored the
other sources under local names. Their configurations were:

| Subset | Source corpus | Config | Text column |
| --- | --- | --- | --- |
| `librispeech_asr_test_clean_eval_1K` | LibriSpeech | `clean` | `text` |
| `voxpopuli_en_eval_1K` | VoxPopuli | `en` | `normalized_text` |
| `tedlium-dev-test_eval_1K` | TED-LIUM | `default` | `text` |
| `gigaspeech_eval_1K` | GigaSpeech | `test` | `text` |
| `spgispeech_S_eval_1K` | SPGISpeech | `S` | `transcript` |
| `earnings22_chunked_eval_1K` | Earnings-22 | `chunked` | `transcription` |
| `ami_ihm_eval_1K` | AMI | `ihm` | `text` |
| `common_voice_17_0_en_eval_1K` | Common Voice 17 | `en` | `sentence` |

Add an entry for each source you can access. To use local subsets in later stages,
list their directories in a dataset config, e.g.
`{"data/subsets/common_voice_17_0_en_eval_1K": {"split": "test", "text_column": "reference"}}`.
Per-model CSV files are keyed by the last path component, so local and Hub copies of
a subset are interchangeable.

| Config | Contents |
| --- | --- |
| `configs/datasets_bench.json` | 10 standard benchmarks used for training |
| `configs/datasets_wild.json` | In-the-wild test sets (never used for training) |
| `configs/datasets_wild_additional.json` | Additional out-of-domain sets |
| `configs/datasets_de.json`, `datasets_de_en.json` | German and German+English sets for Table 2 |
| `configs/models.json` | The 41 ASR models, grouped by loader type |
| `configs/model_profile.json` | Mean WER/CER per model from the paper runs; selects proxies |
| `configs/hyperparameters.json` | Tuned regressor hyperparameters per metric |
| `configs/search_space.json` | Randomized-search grid that produced them |

## Pipeline

### 1. Transcribe

```bash
bash scripts/transcribe.sh --dataset-config configs/datasets_bench.json --output-dir runs/asr
bash scripts/transcribe.sh --dataset-config configs/datasets_wild.json --output-dir runs/asr
```

Without `--models`, this runs every model in `configs/models.json`, one at a time.
Pass `--models openai/whisper-large-v3` to run a single model; its loader type comes
from the config. Each run writes `runs/asr/<model>/<dataset>.csv` with `audio_id`,
`reference`, and `hypothesis`. Finished CSVs are skipped. Interrupted runs resume from
a checkpoint every 100 utterances. Utterances that fail to decode are logged and skipped,
as in the paper runs. Proxy features pair rows by position, so check that a model's
CSV has as many rows as its proxy's.

`--language` defaults to `en` for Whisper. SeamlessM4T expects three-letter codes, so
run it with `--language eng` (or `deu` for German). SpeechBrain checkpoints download
to `--speechbrain-savedir` (default `pretrained_models/`). Use `--device` to choose
the GPU.

### 2. Score

```bash
python -m approx_asr.metrics --root-dir runs/asr
```

This adds per-utterance `WER`, `CER` (percent), word and character error counts
(`nwErrors`, `ncErrors`), their substitution/deletion/insertion breakdown, and
reference lengths (`wRefLength`, `cRefLength`) to every CSV in place. Both texts are
normalized with Whisper's `BasicTextNormalizer` and scored with jiwer.

### 3. SONAR similarity

In the SONAR environment:

```bash
bash scripts/similarity.sh --asr-dir runs/asr --dataset-config configs/datasets_bench.json --output-dir runs/features
bash scripts/similarity.sh --asr-dir runs/asr --dataset-config configs/datasets_wild.json --output-dir runs/features
```

This writes the same rows plus a `similarity` column to `runs/features/<model>/<dataset>.csv`.
For German use `--speech-language deu --text-language deu_Latn`.

### 4. Proxy profile and data splits

`configs/model_profile.json` is the profile used in the paper. To recompute it from
your own outputs:

```bash
python -m approx_asr.proxy --features-dir runs/features --output runs/model_profile.json
```

Then pass `--model-profile runs/model_profile.json` to later stages. For each target
model, the proxy is the model with the lowest mean WER other than the target itself.

Split each benchmark into training (90%) and tuning (10%) rows, with seed 42:

```bash
python -m approx_asr.splits --features-dir runs/features
```

In-the-wild sets are copied whole into `runs/splits/train/`.

### 5. Hyperparameter search (optional)

```bash
python -m approx_asr.tune --metric WER
```

This runs a 100-iteration randomized search (5-fold CV, MAE) over
`configs/search_space.json` on the tuning splits of all models. It writes
`runs/tuning/WER_neg_mean_absolute_error_best_params.json`. The results for WER, CER,
nwErrors, and ncErrors are in `configs/hyperparameters.json`. See the note on tuned
hyperparameters under [Not included](#not-included-and-known-deviations).

### 6. Regression

The regressor standardizes the features. It then stacks a random forest, gradient
boosting, and Poisson histogram gradient boosting under a ridge regressor with
non-negative coefficients.

```bash
bash scripts/regression.sh benchmark --output-dir runs/regression
```

For every model and each of the 10 benchmarks, this trains on the other nine
benchmarks (training and tuning rows) and tests on the held-out benchmark. Each fold's
regressor also predicts the in-the-wild sets. Targets default to `WER,CER,nwErrors,ncErrors`.
The command writes `runs/regression/<model>/<benchmark>_predicted.csv` and
`runs/regression/<model>/<benchmark>/<wild>_predicted.csv`, adding an
`a<metric>` column for each target. Use `--model-ids` to limit the models.

Average the in-the-wild predictions over the 10 folds, then compute corpus-level
results:

```bash
bash scripts/results.sh average-wild --root-dir runs/regression
bash scripts/results.sh summarize --root-dir runs/regression --output-dir runs/results
```

`summary.csv` has one row per model and dataset. Actual WER/CER are corpus-level
rates. aWER/aCER are the predicted word/character error counts (clipped at zero)
divided by the total reference length. `MAE_*` columns compare per-utterance predictions
with the truth. The values come from 500-utterance bootstrap resamples (see
[Not included](#not-included-and-known-deviations)).

Plot actual against approximated WER:

```bash
python -m approx_asr.plots --panel-by model --panels whisper-large-v3 wav2vec2-base-960h \
    --output runs/results/actual_vs_approx_benchmarks.pdf
python -m approx_asr.plots --panel-by dataset --metric CER \
    --panels LibriSpeech_test_noise_eval_1K Primock57_eval_1K \
        atcosim_corpus_eval_1K voxpopuli_en_accented_eval_1K \
    --output runs/results/actual_vs_approx_wild_cer.pdf
```

Dataset names in the summary drop the `_predicted` suffix and keep `_eval_1K`. Use
`--include` to restrict the x axis.

## Reproducing the paper

| Result | Commands |
| --- | --- |
| Figs. 2, 4, 5; per-model tables in the appendix | `regression.sh benchmark`, `results.sh average-wild`, `results.sh summarize`, `python -m approx_asr.plots` |
| Table 2 (cross-lingual) | `regression.sh multilingual`, below |
| Table 3 (ablations) | `scripts/ablation_sweep.sh`, then `results.sh ablation` |
| eWER3 baseline | `scripts/ewer3.sh`, then `results.sh ablation` |
| Table 4 (regression vs. direct proxy) | `results.sh compare-proxy` |
| Fig. 3 (training size) | `regression.sh scaling`, then `results.sh summarize` |
| Table 8 (CMD) | `python -m approx_asr.domain_shift embed`, then `... cmd` |
| Table 9 (TVO) | `python -m approx_asr.domain_shift tvo` |

### Regression vs. direct proxy scoring (Table 4)

```bash
bash scripts/results.sh compare-proxy --root-dir runs/regression --metric nwErrors
```

The direct baseline uses the target hypothesis's error count against the proxy
hypothesis as the estimate. The command reports the MAE of that estimate and of the
regression prediction against the true error count, per model and dataset. It writes
`runs/results/compare_proxy.csv` and prints per-dataset means.

### Training-set size (Fig. 3)

```bash
bash scripts/regression.sh scaling --model-ids data2vec-audio-base-960h --output-dir runs/scaling
bash scripts/results.sh summarize \
    --root-dir runs/scaling/data2vec-audio-base-960h --output-dir runs/results/scaling
```

The first command trains on 1,000 to 10,000 pooled benchmark utterances (seed 42) and
predicts the in-the-wild sets. In the summary, the `model` column holds the training
size. The original script defaulted to `data2vec-audio-base-960h`.

### Cross-lingual transfer (Table 2)

Transcribe the German subsets with `--dataset-config configs/datasets_de.json` for
`openai/whisper-large-v3` (`--language de`) and `facebook/seamless-m4t-v2-large`
(`--language deu`), into `runs/asr_de`. Score them, then compute similarity with the German SONAR
encoder into `runs/features_de`. Copy the two models' English LibriSpeech and Common
Voice CSVs from `runs/features` into the same model directories. Then run:

```bash
bash scripts/regression.sh multilingual --data-dir runs/features_de \
    --model-id seamless-m4t-v2-large --proxy-model-id whisper-large-v3
bash scripts/regression.sh multilingual --data-dir runs/features_de \
    --model-id whisper-large-v3 --proxy-model-id seamless-m4t-v2-large
```

Each command trains on one dataset and tests on each of the others. It writes a JSON
of word-error-count MAEs to `runs/multilingual/`.

### Ablations (Table 3)

```bash
bash scripts/ablation_sweep.sh
bash scripts/results.sh ablation --root-dir runs/ablation --output runs/results/ablation.json
```

The sweep trains on `data2vec-audio-base-960h` and transfers to `wav2vec2-base-960h`
(override with `SOURCE_MODEL` and `OOD_MODEL`). It covers similarity only, proxy only,
both, 2 to 20 proxies (as separate features or averaged), and fixed proxies of
different quality. Each configuration uses 5-fold CV on the pooled benchmark training
splits and predicts word and character error counts. Predictions are written for the
source model's held-out fold (`iid_test`), the same utterances transcribed by the OOD
model, and the in-the-wild sets for both models. `results.sh ablation` reports the
MAE mean and standard deviation over folds.

### eWER3 baseline

```bash
bash scripts/ewer3.sh --root-dir runs/features --apply-pca --n-components 32 --device cuda
bash scripts/ewer3.sh --root-dir runs/features --apply-pca --n-components 64 --device cuda
bash scripts/results.sh ablation --root-dir runs/ewer3 --output runs/results/ewer3.json
```

Our reimplementation of eWER3 (no public code was available) extracts features from
frozen `facebook/wav2vec2-base-960h` and `roberta-base` encoders. It uses the
first-position hidden state of each. PCA reduces each modality to 32 or 64
components, and the result is fed to the same regressor as our method. Evaluation
follows the ablation protocol. `--loss mean_squared_error` switches the ridge
selection criterion.

### Domain shift (Tables 8 and 9)

```bash
python -m approx_asr.domain_shift embed \
    --dataset-config configs/datasets_bench.json configs/datasets_wild.json
python -m approx_asr.domain_shift cmd \
    --dataset-config configs/datasets_bench.json configs/datasets_wild.json
python -m approx_asr.domain_shift tvo \
    --dataset-config configs/datasets_bench.json configs/datasets_wild.json
```

`embed` (SONAR environment) stores one speech embedding per utterance in
`runs/embeddings/`. `cmd` computes the central moment discrepancy between
LibriSpeech-clean and every other set. `tvo` computes the Jaccard overlap between the
lowercased word vocabularies of the reference transcripts. Both write JSON to
`runs/domain_shift/`.

## Layout

| Path | Contents |
| --- | --- |
| `src/approx_asr/subsets.py` | Building and loading the 1K evaluation subsets |
| `src/approx_asr/asr_models.py`, `transcribe.py` | ASR model loaders and batch transcription |
| `src/approx_asr/metrics.py` | Text normalization, WER/CER and error counts |
| `src/approx_asr/similarity.py` | SONAR speech-text similarity |
| `src/approx_asr/proxy.py`, `features.py` | Proxy selection, model profiling, proxy features |
| `src/approx_asr/splits.py`, `tune.py` | Train/tuning splits and hyperparameter search |
| `src/approx_asr/regressor.py` | The stacked regressor |
| `src/approx_asr/experiments.py` | Leave-one-benchmark-out, scaling, cross-lingual runs |
| `src/approx_asr/ablation.py`, `ewer3.py` | Ablations and the eWER3 baseline |
| `src/approx_asr/domain_shift.py` | CMD and TVO |
| `src/approx_asr/results.py`, `plots.py` | Result tables and figures |
| `configs/` | Dataset lists, model list, proxy profile, hyperparameters |
| `scripts/` | Bash launchers |
| `tests/` | CPU tests on synthetic data |

No transcripts, predictions, embeddings, or results are bundled. All stages write
under `runs/`, which Git ignores.

## Not included and known deviations

- **CHiME-6, the perturbed sets, and the medical set.** The paper does not specify how
  the CHiME-6 noisy subset was selected or the parameters of the eight perturbations.
  The medical ASR dataset is private. `configs/datasets_wild_additional.json` still
  lists the perturbed and adversarial subsets by their original Hub IDs, which may not
  be accessible.
- **Source dataset IDs.** Only Common Voice 17 has a recorded source ID (see [Data](#data)).
- **Tuned hyperparameters are not applied by default.** The original pipeline builder
  merged the tuned `stack__*` keys into `rf__*`-style defaults and read only the
  defaults, so the paper's numbers come from the default hyperparameters in
  `regressor.py`. This release keeps that behavior. Pass `--apply-tuned-hyperparameters`
  to `regression.sh benchmark|scaling` or `ablation.sh` to apply
  `configs/hyperparameters.json`.
- **Proxy features in the ablations.** The original ablation script fed the proxy
  columns of every predicted metric to every target, e.g. both `Ref_nwErrors` and
  `Ref_ncErrors` when predicting `nwErrors`. A later rebuttal version used only the
  target's own proxy columns. We have not confirmed which version produced Table 3.
  `--proxy-features all-metrics` (default) keeps the original behavior, and
  `--proxy-features per-metric` gives the rebuttal behavior.
- **Periods in proxy transcripts.** When scoring against the proxy, every `.` in the
  proxy hypothesis is replaced with `no_audio`. This was apparently meant for empty
  hypotheses, but it applies to all sentence punctuation (`Done.` becomes `doneno audio`
  after normalization) and inflates pWER/pCER for proxies that emit punctuation. It is
  kept because it shaped the published features.
- **CMD in Table 8.** It was computed on each dataset's mean SONAR embedding, so every
  higher central moment is zero. The reported CMD is the Euclidean distance between
  mean embeddings. `cmd --per-utterance` computes CMD over all utterance embeddings.
- **Bootstrap in the summaries.** The analysis script reused `random_state=42` in every
  bootstrap iteration, so all 100 iterations draw the same 500-row resample and the
  reported standard deviation is zero. `summarize` keeps this.
  `--independent-resamples` draws a new resample per iteration.
- **Proxy selection** ranks models by mean WER over every subset in the profile, which
  in the paper runs included the in-the-wild sets. Proxy and target rows are matched by
  position within a dataset, not by audio ID.
- **Hyperparameter search.** `configs/search_space.json` is the larger grid that
  contains every value in `configs/hyperparameters.json`. The last version of the search
  script used a smaller grid and rescaled and log-transformed the targets. Neither is
  consistent with the saved values, so both were left out.

## Development

```bash
python -m pip install -e '.[dev]'
ruff check src tests
ruff format --check src tests
pytest
```

## License and attribution

A repository-wide license for this release has not yet been selected. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for adapted code. ASR checkpoints,
SONAR, and the evaluation corpora keep their own licenses and terms of use.

## Citation

```bibtex
@inproceedings{waheed-etal-2025-robust,
    title = "On the Robust Approximation of {ASR} Metrics",
    author = "Waheed, Abdul  and Atwany, Hanin  and Singh, Rita  and Raj, Bhiksha",
    editor = "Che, Wanxiang  and Nabende, Joyce  and Shutova, Ekaterina  and Pilehvar, Mohammad Taher",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.1187/",
    doi = "10.18653/v1/2025.findings-acl.1187",
    pages = "23119--23146",
    ISBN = "979-8-89176-256-5"
}
```
