#!/usr/bin/env bash
# Feature and proxy ablations from the paper's ablation runs (Table 3).
# Extra arguments, e.g. --proxy-features per-metric, are forwarded to every run.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

SOURCE_MODEL="${SOURCE_MODEL:-data2vec-audio-base-960h}"
OOD_MODEL="${OOD_MODEL:-wav2vec2-base-960h}"

run() {
    python -m approx_asr.ablation \
        --model-ids "$SOURCE_MODEL" \
        --ood-model-ids "$OOD_MODEL" \
        "$@"
}

# Similarity and proxy reference, each alone and together.
run --use-similarity --proxy-ref-count 1 "$@"
run --use-similarity --proxy-ref-count 0 "$@"
run --proxy-ref-count 1 "$@"

# Multiple proxy references, as separate features and averaged.
for count in 2 3 4 5 10 20; do
    run --use-similarity --proxy-ref-count "$count" "$@"
done
for count in 3 5 10 20; do
    run --use-similarity --proxy-ref-count "$count" --mean-of-proxies "$@"
done

# Fixed proxies of different quality, with and without similarity.
for proxy in whisper-large-v3 whisper-medium.en whisper-tiny mms-1b-fl102; do
    run --use-similarity --ref-model-id "$proxy" "$@"
    run --ref-model-id "$proxy" "$@"
done
