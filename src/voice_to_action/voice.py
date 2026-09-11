"""Speech-to-text helpers.

All audio dependencies are optional. Typed input always works; if the ``voice`` extra is not
installed a :class:`VoiceUnavailableError` with installation instructions is raised instead.

Backends:
    * ``google``          - SpeechRecognition + Google Web Speech API (online, no API key)
    * ``whisper``         - SpeechRecognition + openai-whisper (offline, needs ``openai-whisper``)
    * ``faster-whisper``  - faster-whisper / CTranslate2 (offline, needs ``faster-whisper``)
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

from voice_to_action import config

VOICE_BACKENDS: tuple[str, ...] = ("google", "whisper", "faster-whisper")
_SR_NATIVE_SUFFIXES = {".wav", ".aiff", ".aif", ".flac"}


class VoiceUnavailableError(RuntimeError):
    """Raised when speech recognition cannot run (missing dependency, no mic, bad audio...)."""


def _require_sr():  # type: ignore[no-untyped-def]
    try:
        import speech_recognition as sr
    except ImportError as exc:
        raise VoiceUnavailableError(
            "Voice input needs the optional 'voice' extra. Install it with: pip install 'voice-to-action[voice]'"
        ) from exc
    return sr


@contextlib.contextmanager
def _materialize(source: str | Path | bytes, filename: str | None) -> Iterator[str]:
    """Yield a filesystem path for ``source`` (writing bytes to a temp file if needed)."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise VoiceUnavailableError(f"Audio file not found: {path}")
        yield str(path)
        return
    suffix = Path(filename).suffix.lower() if filename else ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".wav")
    try:
        tmp.write(source)
        tmp.flush()
        tmp.close()
        yield tmp.name
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp.name)


def _ensure_wav(path: str) -> str:
    """SpeechRecognition only reads WAV/AIFF/FLAC; convert other formats with pydub + ffmpeg."""
    if Path(path).suffix.lower() in _SR_NATIVE_SUFFIXES:
        return path
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise VoiceUnavailableError(
            f"Cannot read '{Path(path).suffix}' audio without pydub + ffmpeg. Upload a WAV file or install the 'voice' extra."
        ) from exc
    wav_path = f"{path}.converted.wav"
    try:
        AudioSegment.from_file(path).export(wav_path, format="wav")
    except Exception as exc:  # ffmpeg missing or corrupt audio
        raise VoiceUnavailableError(f"Audio conversion failed (is ffmpeg installed?): {exc}") from exc
    return wav_path


def _faster_whisper(path: str, model_size: str) -> str:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise VoiceUnavailableError(
            "faster-whisper backend needs: pip install 'voice-to-action[whisper]'"
        ) from exc
    model = WhisperModel(model_size, compute_type="int8")
    segments, _info = model.transcribe(path, vad_filter=True)
    return " ".join(segment.text.strip() for segment in segments).strip()


def _recognize(recognizer, audio, backend: str, language: str, whisper_model: str) -> str:  # type: ignore[no-untyped-def]
    sr = _require_sr()
    try:
        if backend == "whisper":
            return str(recognizer.recognize_whisper(audio, model=whisper_model, language=language.split("-")[0])).strip()
        return str(recognizer.recognize_google(audio, language=language)).strip()
    except sr.UnknownValueError as exc:
        raise VoiceUnavailableError("Could not understand the audio. Try speaking closer to the mic or re-recording.") from exc
    except sr.RequestError as exc:
        raise VoiceUnavailableError(f"Speech service error: {exc}") from exc
    except ImportError as exc:
        raise VoiceUnavailableError("The 'whisper' backend needs: pip install openai-whisper") from exc


def transcribe_audio(
    source: str | Path | bytes,
    backend: str | None = None,
    language: str | None = None,
    whisper_model: str | None = None,
    filename: str | None = None,
) -> str:
    """Transcribe an audio file (path) or raw audio bytes and return the recognised text."""
    backend = (backend or config.VOICE_BACKEND).lower()
    language = language or config.LANGUAGE
    whisper_model = whisper_model or config.WHISPER_MODEL
    if backend not in VOICE_BACKENDS:
        raise ValueError(f"Unknown voice backend {backend!r}; expected one of {VOICE_BACKENDS}")

    with _materialize(source, filename) as path:
        if backend == "faster-whisper":
            return _faster_whisper(path, whisper_model)
        sr = _require_sr()
        wav_path = _ensure_wav(path)
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as audio_file:
                audio = recognizer.record(audio_file)
            return _recognize(recognizer, audio, backend, language, whisper_model)
        finally:
            if wav_path != path:
                with contextlib.suppress(OSError):
                    os.unlink(wav_path)


def transcribe_microphone(
    backend: str | None = None,
    language: str | None = None,
    whisper_model: str | None = None,
    timeout: float = 5.0,
    phrase_time_limit: float = 15.0,
) -> str:
    """Listen on the default microphone for one phrase and transcribe it (needs PyAudio)."""
    backend = (backend or config.VOICE_BACKEND).lower()
    language = language or config.LANGUAGE
    whisper_model = whisper_model or config.WHISPER_MODEL
    sr = _require_sr()
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as mic:
            recognizer.adjust_for_ambient_noise(mic, duration=0.5)
            audio = recognizer.listen(mic, timeout=timeout, phrase_time_limit=phrase_time_limit)
    except (OSError, AttributeError) as exc:
        raise VoiceUnavailableError(
            "Microphone access failed. Install PyAudio (pip install 'voice-to-action[mic]') and check your input device."
        ) from exc
    except sr.WaitTimeoutError as exc:
        raise VoiceUnavailableError("No speech detected before the timeout.") from exc

    if backend == "faster-whisper":
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        try:
            tmp.write(audio.get_wav_data())
            tmp.close()
            return _faster_whisper(tmp.name, whisper_model)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp.name)
    return _recognize(recognizer, audio, backend, language, whisper_model)
