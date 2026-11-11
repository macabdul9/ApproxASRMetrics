"""Loaders and single-utterance transcription for every model family in configs/models.json.

Decoding is greedy with each library's defaults. Model types match the keys of
configs/models.json.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import (
    AutoModel,
    AutoProcessor,
    Data2VecAudioForCTC,
    HubertForCTC,
    MoonshineForConditionalGeneration,
    SeamlessM4TModel,
    SeamlessM4Tv2Model,
    Wav2Vec2ConformerForCTC,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

SAMPLING_RATE = 16000


@dataclass
class LoadedModel:
    model: object
    processor: object | None = None


def _hf(model_cls, processor_cls=AutoProcessor):
    def load(model_name: str, device: str, **_) -> LoadedModel:
        model = model_cls.from_pretrained(model_name).to(device).eval()
        return LoadedModel(model, processor_cls.from_pretrained(model_name))

    return load


def _with_temp_wav(audio: np.ndarray, transcribe: Callable[[str], list]) -> list:
    """Run a NeMo `transcribe` that needs a file path on a temporary 16 kHz WAV."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        path = handle.name
    try:
        sf.write(path, audio, SAMPLING_RATE)
        return transcribe(path)
    finally:
        os.remove(path)


@torch.no_grad()
def transcribe_ctc(loaded: LoadedModel, audio: np.ndarray, language: str | None) -> str:
    inputs = loaded.processor(
        audio, return_tensors="pt", padding="longest", sampling_rate=SAMPLING_RATE
    ).input_values.to(loaded.model.device)
    predicted_ids = torch.argmax(loaded.model(inputs).logits, dim=-1)
    return loaded.processor.batch_decode(predicted_ids)[0]


@torch.no_grad()
def transcribe_hubert(loaded: LoadedModel, audio: np.ndarray, language: str | None) -> str:
    inputs = loaded.processor(
        audio, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True
    ).to(loaded.model.device)
    predicted_ids = torch.argmax(loaded.model(inputs.input_values).logits, dim=-1)
    return loaded.processor.batch_decode(predicted_ids)[0]


@torch.no_grad()
def transcribe_seq2seq(loaded: LoadedModel, audio: np.ndarray, language: str | None) -> str:
    """Whisper and Moonshine; `language` is passed to generate only when given."""
    inputs = loaded.processor(audio, return_tensors="pt", sampling_rate=SAMPLING_RATE).to(
        loaded.model.device
    )
    kwargs = {"language": language} if language else {}
    outputs = loaded.model.generate(**inputs, **kwargs)
    return loaded.processor.decode(outputs[0], skip_special_tokens=True)


@torch.no_grad()
def transcribe_seamless(loaded: LoadedModel, audio: np.ndarray, language: str | None) -> str:
    """SeamlessM4T needs a three-letter target language such as `eng` or `deu`."""
    inputs = loaded.processor(audios=audio, sampling_rate=SAMPLING_RATE, return_tensors="pt").to(
        loaded.model.device
    )
    outputs = loaded.model.generate(**inputs, tgt_lang=language or "eng", generate_speech=False)
    return loaded.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]


def load_speechllm(model_name: str, device: str, **_) -> LoadedModel:
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    return LoadedModel(model)


@torch.no_grad()
def transcribe_speechllm(loaded: LoadedModel, audio: np.ndarray, language: str | None) -> str:
    output = loaded.model.generate_meta(
        audio_tensor=torch.tensor(audio).unsqueeze(0).to(loaded.model.device),
        instruction="Give me the following information about the audio [Transcript]",
        max_new_tokens=500,
    )
    # The model answers with a JSON-like string; the transcript is the second quoted value.
    return output.split('"')[3]


def load_speechbrain(model_name: str, device: str, savedir: Path, **_) -> LoadedModel:
    from speechbrain.inference.ASR import EncoderASR

    model = EncoderASR.from_hparams(
        source=model_name,
        savedir=str(savedir / model_name.split("/")[-1]),
        run_opts={"device": device},
    )
    return LoadedModel(model)


def transcribe_speechbrain(loaded: LoadedModel, audio: np.ndarray, language: str | None) -> str:
    transcriptions, _ = loaded.model.transcribe_batch(
        wavs=torch.tensor(audio).unsqueeze(0), wav_lens=torch.tensor([1.0])
    )
    return transcriptions[0]


def load_canary(model_name: str, device: str, **_) -> LoadedModel:
    from nemo.collections.asr.models import EncDecMultiTaskModel

    model = EncDecMultiTaskModel.from_pretrained(model_name).to(device).eval()
    # Greedy decoding, as recommended on the model card.
    decoding = model.cfg.decoding
    decoding.beam.beam_size = 1
    model.change_decoding_strategy(decoding)
    return LoadedModel(model)


def load_nemo_rnnt(model_name: str, device: str, **_) -> LoadedModel:
    from nemo.collections.asr.models import EncDecRNNTBPEModel

    return LoadedModel(EncDecRNNTBPEModel.from_pretrained(model_name=model_name).to(device).eval())


def load_nemo_ctc(model_name: str, device: str, **_) -> LoadedModel:
    from nemo.collections.asr.models import EncDecCTCModelBPE

    return LoadedModel(EncDecCTCModelBPE.from_pretrained(model_name=model_name).to(device).eval())


def load_nemo_asr(model_name: str, device: str, **_) -> LoadedModel:
    from nemo.collections.asr.models import ASRModel

    return LoadedModel(ASRModel.from_pretrained(model_name=model_name))


def transcribe_nemo_file(loaded: LoadedModel, audio: np.ndarray, language: str | None) -> str:
    return _with_temp_wav(audio, lambda path: loaded.model.transcribe([path]))[0].text.strip()


def transcribe_nemo_ctc(loaded: LoadedModel, audio: np.ndarray, language: str | None) -> str:
    return loaded.model.transcribe(audio, batch_size=1, verbose=False)[0]


def transcribe_nemo_asr(loaded: LoadedModel, audio: np.ndarray, language: str | None) -> str:
    greedy, _ = loaded.model.transcribe(audio, batch_size=1, verbose=False)
    return greedy[0]


def _without_language(transcribe):
    return lambda loaded, audio, language: transcribe(loaded, audio, None)


MODEL_TYPES = {
    "wav2vec2": (_hf(Wav2Vec2ForCTC, Wav2Vec2Processor), transcribe_ctc),
    "mms": (_hf(Wav2Vec2ForCTC, Wav2Vec2Processor), transcribe_ctc),
    "wav2vec2_conformer": (_hf(Wav2Vec2ConformerForCTC, Wav2Vec2Processor), transcribe_ctc),
    "data2vec": (_hf(Data2VecAudioForCTC, Wav2Vec2Processor), transcribe_ctc),
    "hubert": (_hf(HubertForCTC, Wav2Vec2Processor), transcribe_hubert),
    "whisper": (_hf(WhisperForConditionalGeneration, WhisperProcessor), transcribe_seq2seq),
    "moonshine": (_hf(MoonshineForConditionalGeneration), _without_language(transcribe_seq2seq)),
    "seamless": (_hf(SeamlessM4TModel), transcribe_seamless),
    "seamlessv2": (_hf(SeamlessM4Tv2Model), transcribe_seamless),
    "speechllm": (load_speechllm, transcribe_speechllm),
    "speechbrain": (load_speechbrain, transcribe_speechbrain),
    "canary": (load_canary, transcribe_nemo_file),
    "encodec_rnnt_bpe": (load_nemo_rnnt, transcribe_nemo_file),
    "encodec_bpe_ctc": (load_nemo_ctc, transcribe_nemo_ctc),
    "nemo_asr": (load_nemo_asr, transcribe_nemo_asr),
    # English-only Whisper checkpoints are decoded without a language token.
    "whisper_en": (
        _hf(WhisperForConditionalGeneration, WhisperProcessor),
        _without_language(transcribe_seq2seq),
    ),
}
