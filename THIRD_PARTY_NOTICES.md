# Third-party attribution

`normalize_text` in `src/approx_asr/metrics.py` copies the default behavior of
`BasicTextNormalizer` from Hugging Face Transformers
(`transformers/models/whisper/english_normalizer.py`, Apache-2.0). That class is in
turn adapted from OpenAI Whisper (MIT). The copy avoids importing Transformers for
scoring. `tests/test_metrics.py` compares it with the Transformers implementation
when Transformers is installed.

`central_moment_discrepancy` in `src/approx_asr/domain_shift.py` reimplements the
central moment discrepancy of Zellinger et al., "Central Moment Discrepancy (CMD) for
Domain-Invariant Representation Learning" (ICLR 2017). The original experiment code
was taken from https://github.com/wzell/mann. The NumPy version here computes the
same quantity without range scaling.

The eWER3 baseline in `src/approx_asr/ewer3.py` is our reimplementation of
Chowdhury and Ali, "Multilingual Word Error Rate Estimation: e-WER3" (ICASSP 2023).
No code from the original authors was used.

Inference relies on PyTorch, Transformers, NVIDIA NeMo, SpeechBrain, and Meta's SONAR
and fairseq2. Their licenses apply independently, as do the terms of each ASR
checkpoint and evaluation corpus.
