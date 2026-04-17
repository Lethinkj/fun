from __future__ import annotations

import asyncio
import struct

from pydub import AudioSegment


def _calculate_rms(pcm_bytes: bytes, sample_width: int = 2) -> int:
    if not pcm_bytes or sample_width <= 0:
        return 0
    num_samples = len(pcm_bytes) // sample_width
    if num_samples == 0:
        return 0

    total = 0
    for i in range(num_samples):
        if sample_width == 2:
            sample = struct.unpack_from("<h", pcm_bytes, i * 2)[0]
        else:
            sample = 0
        total += sample * sample

    return int((total / num_samples) ** 0.5)
import contextlib
import json
import logging
import os
import random
import re
import shutil
import threading
import time
import wave
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import aiohttp
import discord
from discord.ext import commands
from dotenv import load_dotenv
from aiohttp import web

try:
    import edge_tts
except ImportError:  # pragma: no cover - handled at runtime
    edge_tts = None

try:
    from piper import PiperVoice
except ImportError:  # pragma: no cover - handled at runtime
    PiperVoice = None

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - handled at runtime
    WhisperModel = None

try:
    from discord.ext import voice_recv
except ImportError:  # pragma: no cover - handled at runtime
    voice_recv = None

load_dotenv()

AI_API_BASE_URL = os.getenv("AI_API_BASE_URL", "https://openrouter.ai/api/v1")
AI_API_KEY = os.getenv("AI_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))
AI_MODEL = os.getenv("AI_MODEL", os.getenv("OPENROUTER_MODEL", ""))
AI_SITE_URL = os.getenv("AI_SITE_URL", "")
AI_APP_NAME = os.getenv("AI_APP_NAME", "Bee Voice Bot")
AI_TEMPERATURE = float(os.getenv("AI_TEMPERATURE", "0.4"))
AI_MAX_TOKENS = int(os.getenv("AI_MAX_TOKENS", "2000"))
AI_REQUEST_TIMEOUT_SECONDS = float(os.getenv("AI_REQUEST_TIMEOUT_SECONDS", "35"))
AI_RETRY_ATTEMPTS = int(os.getenv("AI_RETRY_ATTEMPTS", "3"))
ASR_BACKEND = os.getenv("ASR_BACKEND", "auto").strip().lower()
LOCAL_ASR_MODEL = os.getenv("LOCAL_ASR_MODEL", "tiny.en").strip()
LOCAL_ASR_DEVICE = os.getenv("LOCAL_ASR_DEVICE", "cpu").strip()
LOCAL_ASR_COMPUTE_TYPE = os.getenv("LOCAL_ASR_COMPUTE_TYPE", "int8").strip()
LIVE_VOICE_ENABLED = os.getenv("LIVE_VOICE_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
LIVE_REQUIRE_WAKE_WORD = os.getenv("LIVE_REQUIRE_WAKE_WORD", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LIVE_TRANSCRIBE_SILENCE_SECONDS = float(os.getenv("LIVE_TRANSCRIBE_SILENCE_SECONDS", "1.0"))
LIVE_TRANSCRIBE_MIN_SECONDS = float(os.getenv("LIVE_TRANSCRIBE_MIN_SECONDS", "0.8"))
LIVE_TRANSCRIBE_MAX_SECONDS = float(os.getenv("LIVE_TRANSCRIBE_MAX_SECONDS", "12"))
LIVE_TRANSCRIBE_LANGUAGE = os.getenv("LIVE_TRANSCRIBE_LANGUAGE", "auto").strip().lower()
LIVE_TEXT_FEEDBACK = os.getenv("LIVE_TEXT_FEEDBACK", "false").strip().lower() in {"1", "true", "yes", "on"}
MIN_RMS_FOR_TRANSCRIPTION = int(os.getenv("MIN_RMS_FOR_TRANSCRIPTION", "220"))

TTS_BACKEND = os.getenv("TTS_BACKEND", "auto").lower()
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AriaNeural")
TAMIL_TTS_VOICE = os.getenv("TAMIL_TTS_VOICE", "ta-IN-PallaviNeural")
TANGLISH_TTS_VOICE = os.getenv("TANGLISH_TTS_VOICE", TAMIL_TTS_VOICE)
TTS_RATE = os.getenv("TTS_RATE", "+20%")
MAX_TTS_CHARS = int(os.getenv("MAX_TTS_CHARS", "2000"))
VOICE_REPLY_MODE = os.getenv("VOICE_REPLY_MODE", "brief").lower()
VOICE_MAX_CHARS = int(os.getenv("VOICE_MAX_CHARS", "260"))
VOICE_LANGUAGE_MODE = os.getenv("VOICE_LANGUAGE_MODE", "auto").lower()
AI_STYLE = os.getenv("AI_STYLE", "normal").strip().lower()
LOCAL_TTS_MODEL_PATH = Path(os.getenv("LOCAL_TTS_MODEL_PATH", "voices/en_US-lessac-medium.onnx"))
PIPER_USE_CUDA = os.getenv("PIPER_USE_CUDA", "false").strip().lower() in {"1", "true", "yes", "on"}
TANGLISH_PREFER_LOCAL_PIPER = os.getenv("TANGLISH_PREFER_LOCAL_PIPER", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

MENTION_TRIGGER = os.getenv("MENTION_TRIGGER", "@fun").strip().lower()
MENTION_KEYWORD = MENTION_TRIGGER.lstrip("@")
WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("PORT", os.getenv("WEB_PORT", "10000")))
DISCORD_LOGIN_MAX_RETRIES = int(os.getenv("DISCORD_LOGIN_MAX_RETRIES", "8"))
DISCORD_LOGIN_RETRY_BASE_SECONDS = float(os.getenv("DISCORD_LOGIN_RETRY_BASE_SECONDS", "5"))
DISCORD_LOGIN_RETRY_MAX_SECONDS = float(os.getenv("DISCORD_LOGIN_RETRY_MAX_SECONDS", "300"))

log = logging.getLogger(__name__)
_http_session: Optional[aiohttp.ClientSession] = None
_local_tts_voice = None
_local_tts_lock = threading.Lock()
_asr_model = None
_asr_model_lock = threading.Lock()
_opus_decode_guard_applied = False
_discord_login_attempts = 0
_discord_last_error: Optional[str] = None


@dataclass
class GuildSession:
    text_channel_id: int
    speak_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_lyrics_user_id: Optional[int] = None
    recent_language_by_user: dict[int, tuple[str, float]] = field(default_factory=dict)
    chat_history: list[dict[str, str]] = field(default_factory=list)
    listening_enabled: bool = False
    voice_sink: Any = None
    voice_poll_task: Optional[asyncio.Task[Any]] = None
    live_audio_by_user: dict[int, dict[str, Any]] = field(default_factory=dict)
    live_audio_lock: threading.Lock = field(default_factory=threading.Lock)


sessions: dict[int, GuildSession] = {}

intents = discord.Intents.default()
intents.guilds = True
intents.message_content = True
intents.voice_states = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)

TAMIL_CHAR_RE = re.compile(r"[\u0B80-\u0BFF]")
TANGLISH_HINTS = {
    "vanakkam",
    "vannakam",
    "enna",
    "ennada",
    "ennanga",
    "epdi",
    "eppadi",
    "iruka",
    "irukka",
    "unga",
    "ungal",
    "un",
    "nee",
    "neenga",
    "yaru",
    "yaaru",
    "nanri",
    "romba",
    "nalla",
    "iruken",
    "irukken",
    "irukaen",
    "iruka",
    "irukka",
    "irukiya",
    "irukiyaa",
    "seri",
    "sari",
    "venum",
    "venda",
    "ponga",
    "po",
    "vaa",
    "va",
    "da",
    "dei",
    "dai",
    "saptiya",
    "iruku",
    "ille",
    "illa",
    "ama",
    "aama",
    "illai",
    "machi",
    "bro",
    "thala",
    "appo",
    "ippo",
    "thambi",
    "akka",
    "anna",
}

LANGUAGE_STICKY_SECONDS = float(os.getenv("LANGUAGE_STICKY_SECONDS", "300"))
BRIEF_REPLY_MAX_TOKENS = int(os.getenv("BRIEF_REPLY_MAX_TOKENS", "220"))
CHAT_HISTORY_MAX_TURNS = int(os.getenv("CHAT_HISTORY_MAX_TURNS", "6"))
CHAT_HISTORY_MAX_CHARS_PER_MESSAGE = int(os.getenv("CHAT_HISTORY_MAX_CHARS_PER_MESSAGE", "220"))
PERSIST_CHAT_MEMORY = os.getenv("PERSIST_CHAT_MEMORY", "true").strip().lower() in {"1", "true", "yes", "on"}
CHAT_MEMORY_FILE = Path(os.getenv("CHAT_MEMORY_FILE", "recordings/chat_memory.json"))
FULL_SONGS_DIR = Path(os.getenv("FULL_SONGS_DIR", "recordings/fullsongs"))
FULL_SONG_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
ENABLE_LOCAL_LIBRARY = os.getenv("ENABLE_LOCAL_LIBRARY", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
JAMENDO_CLIENT_ID = os.getenv("JAMENDO_CLIENT_ID", "")
ALLOW_PREVIEW_FALLBACK = os.getenv("ALLOW_PREVIEW_FALLBACK", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def configure_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    has_file_handler = any(
        isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "").endswith("bot.log")
        for h in root.handlers
    )
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
        for h in root.handlers
    )
    if has_file_handler and has_stream_handler:
        return

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    if not has_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    if not has_file_handler:
        file_handler = RotatingFileHandler("bot.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


configure_logging()


def get_local_tts_voice() -> Any:
    if PiperVoice is None:
        raise RuntimeError("Local TTS is not installed. Install piper-tts first.")

    global _local_tts_voice
    with _local_tts_lock:
        if _local_tts_voice is None:
            if not LOCAL_TTS_MODEL_PATH.exists():
                raise RuntimeError(f"Local TTS model not found: {LOCAL_TTS_MODEL_PATH}")
            _local_tts_voice = PiperVoice.load(LOCAL_TTS_MODEL_PATH, use_cuda=PIPER_USE_CUDA)
        return _local_tts_voice


def resolve_tts_backend() -> str:
    if TTS_BACKEND in {"piper", "edge"}:
        return TTS_BACKEND
    if LOCAL_TTS_MODEL_PATH.exists() and PiperVoice is not None:
        return "piper"
    return "edge"


def get_asr_model() -> Any:
    if WhisperModel is None:
        raise RuntimeError("faster-whisper is not installed. Install it to enable live voice listening.")

    global _asr_model
    with _asr_model_lock:
        if _asr_model is None:
            _asr_model = WhisperModel(
                LOCAL_ASR_MODEL,
                device=LOCAL_ASR_DEVICE,
                compute_type=LOCAL_ASR_COMPUTE_TYPE,
            )
        return _asr_model


def _pcm_bytes_to_seconds(pcm_bytes: bytes, sample_rate: int = 48_000, channels: int = 2, sample_width: int = 2) -> float:
    frame_bytes = sample_width * channels
    if frame_bytes <= 0:
        return 0.0
    frames = len(pcm_bytes) / frame_bytes
    return frames / float(sample_rate)


def transcribe_pcm_bytes(pcm_bytes: bytes) -> str:
    if not pcm_bytes:
        return ""

    # Incoming PCM from discord-ext-voice-recv is 48kHz, s16le, stereo.
    audio = AudioSegment(
        data=pcm_bytes,
        sample_width=2,
        frame_rate=48_000,
        channels=2,
    )
    audio = audio.set_channels(1).set_frame_rate(16_000)

    asr_dir = Path("recordings") / "asr"
    asr_dir.mkdir(parents=True, exist_ok=True)
    wav_path = asr_dir / f"asr_{int(time.time() * 1000)}.wav"

    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(audio.raw_data)

    try:
        model = get_asr_model()
        language = None if LIVE_TRANSCRIBE_LANGUAGE in {"", "auto"} else LIVE_TRANSCRIBE_LANGUAGE
        segments, _info = model.transcribe(
            str(wav_path),
            language=language,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text and segment.text.strip())
        return text.strip()
    finally:
        wav_path.unlink(missing_ok=True)


def apply_opus_decode_guard() -> None:
    global _opus_decode_guard_applied
    if _opus_decode_guard_applied:
        return

    decoder_cls = getattr(discord.opus, "Decoder", None)
    opus_error_cls = getattr(discord.opus, "OpusError", Exception)
    if decoder_cls is None or not hasattr(decoder_cls, "decode"):
        return

    original_decode = decoder_cls.decode

    def _safe_decode(self: Any, data: Any, *, fec: bool = False) -> bytes:
        try:
            return original_decode(self, data, fec=fec)
        except opus_error_cls:
            # Corrupted RTP/Opus frames happen on lossy links; return silence to keep recv alive.
            channels = int(getattr(decoder_cls, "CHANNELS", 2))
            frame_samples = int(getattr(decoder_cls, "SAMPLES_PER_FRAME", 960))
            sample_width = 2  # s16le
            return b"\x00" * (frame_samples * channels * sample_width)

    decoder_cls.decode = _safe_decode
    _opus_decode_guard_applied = True
    log.info("Applied Opus decode guard for live voice receive")


async def get_http_session() -> aiohttp.ClientSession:
    global _http_session
    if _http_session is None or _http_session.closed:
        timeout = aiohttp.ClientTimeout(total=45, connect=10)
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=25, ttl_dns_cache=300)
        _http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    return _http_session


async def search_song_preview(query: str) -> Optional[dict[str, str]]:
    session = await get_http_session()
    url = "https://itunes.apple.com/search"
    params = {
        "term": query,
        "media": "music",
        "entity": "song",
        "limit": "1",
    }
    async with session.get(url, params=params, timeout=15) as resp:
        if resp.status >= 400:
            return None
        payload = await resp.json(content_type=None)

    results = payload.get("results") or []
    if not results:
        return None

    top = results[0]
    preview_url = top.get("previewUrl")
    if not preview_url:
        return None

    return {
        "track": str(top.get("trackName") or "Unknown Track"),
        "artist": str(top.get("artistName") or "Unknown Artist"),
        "preview_url": str(preview_url),
    }


async def search_jamendo_full_track(query: str) -> Optional[dict[str, str]]:
    if not JAMENDO_CLIENT_ID:
        return None

    session = await get_http_session()
    url = "https://api.jamendo.com/v3.0/tracks/"
    params = {
        "client_id": JAMENDO_CLIENT_ID,
        "format": "json",
        "limit": "1",
        "audioformat": "mp32",
        "search": query,
    }
    async with session.get(url, params=params, timeout=15) as resp:
        if resp.status >= 400:
            return None
        payload = await resp.json(content_type=None)

    results = payload.get("results") or []
    if not results:
        return None

    top = results[0]
    audio_url = top.get("audio")
    if not audio_url:
        return None

    return {
        "track": str(top.get("name") or "Unknown Track"),
        "artist": str(top.get("artist_name") or "Unknown Artist"),
        "audio_url": str(audio_url),
    }


def search_local_full_song(query: str) -> Optional[Path]:
    if not ENABLE_LOCAL_LIBRARY:
        return None

    if not FULL_SONGS_DIR.exists() or not FULL_SONGS_DIR.is_dir():
        return None

    normalized_query = re.sub(r"[^a-z0-9]+", " ", query.lower()).strip()
    if not normalized_query:
        return None

    candidates = []
    for path in FULL_SONGS_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in FULL_SONG_EXTENSIONS:
            continue
        stem = re.sub(r"[^a-z0-9]+", " ", path.stem.lower()).strip()
        if normalized_query == stem:
            return path
        if normalized_query in stem:
            candidates.append(path)

    return sorted(candidates)[0] if candidates else None


def is_direct_audio_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"}:
        return False
    lowered_path = parsed.path.lower()
    return any(lowered_path.endswith(ext) for ext in FULL_SONG_EXTENSIONS)


async def health_handler(_request: web.Request) -> web.Response:
    return web.json_response(
        {
            "status": "ok",
            "bot_ready": bot.is_ready(),
            "bot_user": str(bot.user) if bot.user else None,
            "active_voice_sessions": len(sessions),
            "discord_login_attempts": _discord_login_attempts,
            "discord_last_error": _discord_last_error,
        }
    )


async def start_healthcheck_server() -> web.AppRunner:
    app = web.Application()
    app.router.add_get("/", health_handler)
    app.router.add_get("/health", health_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=WEB_HOST, port=WEB_PORT)
    await site.start()
    log.info("Healthcheck server listening on %s:%s", WEB_HOST, WEB_PORT)
    return runner


async def ensure_voice_client(ctx: commands.Context) -> Optional[discord.VoiceClient]:
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("Join a voice channel first.")
        return None

    channel = ctx.author.voice.channel
    current = ctx.voice_client

    if current and current.channel.id != channel.id:
        await current.move_to(channel)

    if not ctx.voice_client:
        try:
            connect_kwargs: dict[str, Any] = {"timeout": 20}
            if LIVE_VOICE_ENABLED and voice_recv is not None:
                connect_kwargs["cls"] = voice_recv.VoiceRecvClient
            vc = await channel.connect(**connect_kwargs)
        except asyncio.TimeoutError:
            await ctx.send("Failed to connect (timeout).")
            return None
    else:
        vc = ctx.voice_client

    return vc


def detect_prompt_language(prompt: str, session_state: Optional[GuildSession] = None, author_id: Optional[int] = None) -> str:
    mode = VOICE_LANGUAGE_MODE
    if mode in {"en", "english"}:
        return "en"
    if mode in {"ta", "tamil"}:
        return "ta"
    if mode in {"ta-latn", "tanglish", "tunglish"}:
        return "ta-latn"

    if TAMIL_CHAR_RE.search(prompt):
        if session_state and author_id is not None:
            session_state.recent_language_by_user[author_id] = ("ta", time.time())
        return "ta"

    words = re.findall(r"[a-z']+", prompt.lower())
    hint_count = sum(1 for word in words if word in TANGLISH_HINTS)
    if hint_count >= 2:
        if session_state and author_id is not None:
            session_state.recent_language_by_user[author_id] = ("ta-latn", time.time())
        return "ta-latn"

    # Catch short common Tanglish questions such as "yaru nee", "epdi iruka", etc.
    if words and len(words) <= 5 and hint_count >= 1:
        if session_state and author_id is not None:
            session_state.recent_language_by_user[author_id] = ("ta-latn", time.time())
        return "ta-latn"

    # Keep short follow-up chat in the same detected language for a short window.
    if session_state and author_id is not None:
        prior = session_state.recent_language_by_user.get(author_id)
        if prior:
            prior_lang, prior_ts = prior
            if (time.time() - prior_ts) <= LANGUAGE_STICKY_SECONDS and prior_lang in {"ta", "ta-latn"}:
                session_state.recent_language_by_user[author_id] = (prior_lang, time.time())
                return prior_lang

    return "en"


def build_system_instruction(language_code: str) -> str:
    base = (
        "You are a Discord voice assistant. Give clear, well-structured responses with good grammar. "
        "Keep default replies concise for fast speech, and provide more detail only when explicitly asked. "
        "Reply directly to the user. Do not explain what language they used, do not narrate intent, and do not translate unless asked. "
        "You can answer general knowledge, world facts, people, history, science, and everyday questions directly. "
        "Do not claim you are limited to Discord-server-only user data unless the user specifically asks for private Discord records you cannot access."
    )
    style = ""
    if AI_STYLE == "savage":
        style = (
            " Keep a playful savage tone: witty and confident, not aggressive. "
            "Use light roasting only when it fits, keep it short, and avoid repeated insults. "
            "Avoid slurs, hate, threats, harassment, sexual content, or humiliating language. "
            "If the user seems upset or the topic is sensitive, switch to calm respectful style immediately."
        )
    if language_code == "ta":
        return f"{base}{style} Reply in natural Tamil using Tamil script."
    if language_code == "ta-latn":
        return (
            f"{base}{style} Reply in Tanglish/Tunglish (Tamil written in English letters), "
            "easy to read aloud naturally."
        )
    return f"{base}{style} Reply in English."


def _load_chat_memory() -> dict[int, list[dict[str, str]]]:
    if not PERSIST_CHAT_MEMORY or not CHAT_MEMORY_FILE.exists():
        return {}

    try:
        data = json.loads(CHAT_MEMORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning("Failed to load chat memory file: %s", CHAT_MEMORY_FILE)
        return {}

    result: dict[int, list[dict[str, str]]] = {}
    for guild_id, history in data.items():
        try:
            gid = int(guild_id)
        except (TypeError, ValueError):
            continue
        if not isinstance(history, list):
            continue

        cleaned_history: list[dict[str, str]] = []
        for turn in history[-max(1, CHAT_HISTORY_MAX_TURNS * 2):]:
            if not isinstance(turn, dict):
                continue
            user = str(turn.get("user", ""))
            assistant = str(turn.get("assistant", ""))
            speaker = str(turn.get("speaker", ""))
            if user or assistant:
                cleaned_history.append({"speaker": speaker, "user": user, "assistant": assistant})

        if cleaned_history:
            result[gid] = cleaned_history

    return result


def _save_chat_memory() -> None:
    if not PERSIST_CHAT_MEMORY:
        return

    CHAT_MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        str(guild_id): session_state.chat_history[-max(1, CHAT_HISTORY_MAX_TURNS * 2):]
        for guild_id, session_state in sessions.items()
        if session_state.chat_history
    }
    try:
        CHAT_MEMORY_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        log.warning("Failed to save chat memory file: %s", CHAT_MEMORY_FILE)


def clip_for_history(text: str) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= CHAT_HISTORY_MAX_CHARS_PER_MESSAGE:
        return cleaned
    return cleaned[: CHAT_HISTORY_MAX_CHARS_PER_MESSAGE - 3].rstrip() + "..."


def get_recent_chat_context(session_state: GuildSession) -> str:
    history = session_state.chat_history
    if not history:
        return ""

    turns = history[-CHAT_HISTORY_MAX_TURNS:]
    lines = []
    for turn in turns:
        speaker = turn.get("speaker", "").strip()
        user_text = turn.get("user", "").strip()
        assistant_text = turn.get("assistant", "").strip()
        if user_text:
            prefix = speaker or "User"
            lines.append(f"{prefix}: {clip_for_history(user_text)}")
        if assistant_text:
            lines.append(f"Assistant: {clip_for_history(assistant_text)}")

    return "\n".join(lines)


def append_chat_history(session_state: GuildSession, speaker_name: str, user_text: str, assistant_text: str) -> None:
    history = session_state.chat_history
    history.append({
        "speaker": speaker_name,
        "user": user_text,
        "assistant": assistant_text,
    })
    # Keep bounded memory to avoid growth over long uptime.
    max_items = max(1, CHAT_HISTORY_MAX_TURNS * 2)
    if len(history) > max_items:
        del history[:-max_items]
    _save_chat_memory()


def _wake_words() -> list[str]:
    raw = os.getenv("WAKE_WORDS", "")
    return [word.strip().lower() for word in raw.split(",") if word.strip()]


def _passes_wake_word(text: str) -> bool:
    if not LIVE_REQUIRE_WAKE_WORD:
        return True
    lowered = text.lower()
    words = _wake_words()
    if not words:
        return True
    return any(word in lowered for word in words)


def _push_live_audio_frame(session_state: GuildSession, user: Optional[discord.abc.User], pcm: bytes) -> None:
    if not user or not pcm:
        return
    if bot.user and user.id == bot.user.id:
        return

    try:
        rms = _calculate_rms(pcm, 2)
    except Exception:
        rms = 0

    # Ignore near-silence frames so pause detection can flush and trigger a response.
    if rms < MIN_RMS_FOR_TRANSCRIPTION:
        return

    now = time.time()
    with session_state.live_audio_lock:
        slot = session_state.live_audio_by_user.get(user.id)
        if slot is None:
            slot = {
                "user": user,
                "pcm": bytearray(),
                "started_at": now,
                "last_frame_at": now,
            }
            session_state.live_audio_by_user[user.id] = slot

        slot["user"] = user
        slot["last_frame_at"] = now
        casted = slot["pcm"]
        if isinstance(casted, bytearray):
            casted.extend(pcm)


async def _process_live_utterance(
    guild: discord.Guild,
    session_state: GuildSession,
    vc: discord.VoiceClient,
    user: discord.abc.User,
    pcm_bytes: bytes,
) -> None:
    duration_seconds = _pcm_bytes_to_seconds(pcm_bytes)
    if duration_seconds < LIVE_TRANSCRIBE_MIN_SECONDS:
        return

    if duration_seconds > LIVE_TRANSCRIBE_MAX_SECONDS:
        max_bytes = int(LIVE_TRANSCRIBE_MAX_SECONDS * 48_000 * 2 * 2)
        pcm_bytes = pcm_bytes[-max_bytes:]

    transcript = (await asyncio.to_thread(transcribe_pcm_bytes, pcm_bytes)).strip()
    log.info("Live voice transcript: %r", transcript)
    if not transcript:
        return

    if not _passes_wake_word(transcript):
        return

    language_code = detect_prompt_language(transcript, session_state=session_state, author_id=user.id)
    context = get_recent_chat_context(session_state)
    speaker_name = getattr(user, "display_name", None) or getattr(user, "name", "User")

    try:
        response = await generate_ai_reply(
            transcript,
            guild,
            user,
            language_code,
            conversation_context=context,
        )
    except Exception:
        log.exception("Live voice AI request failed transcript=%r", transcript)
        response = "I had a brief AI hiccup. Ask again and I will clap back properly."

    append_chat_history(session_state, str(speaker_name), transcript, response)

    text_channel = guild.get_channel(session_state.text_channel_id)
    if LIVE_TEXT_FEEDBACK and isinstance(text_channel, discord.TextChannel):
        await text_channel.send(f"{speaker_name}: {transcript}\n{response}")

    await speak_text(vc, session_state, response, language_code)


async def _live_voice_poll_loop(guild_id: int) -> None:
    while True:
        session_state = sessions.get(guild_id)
        guild = bot.get_guild(guild_id)
        vc = guild.voice_client if guild else None

        if not session_state or not guild or not vc or not vc.is_connected() or not session_state.listening_enabled:
            return

        ready: list[tuple[discord.abc.User, bytes]] = []
        now = time.time()
        with session_state.live_audio_lock:
            remove_user_ids: list[int] = []
            for user_id, slot in session_state.live_audio_by_user.items():
                user = slot.get("user")
                pcm_buffer = slot.get("pcm")
                last_frame_at = float(slot.get("last_frame_at", now))
                started_at = float(slot.get("started_at", now))

                if not user or not isinstance(pcm_buffer, bytearray):
                    remove_user_ids.append(user_id)
                    continue

                utterance_age = now - started_at
                silence_age = now - last_frame_at
                should_flush = silence_age >= LIVE_TRANSCRIBE_SILENCE_SECONDS or utterance_age >= LIVE_TRANSCRIBE_MAX_SECONDS
                if not should_flush:
                    continue

                ready.append((user, bytes(pcm_buffer)))
                remove_user_ids.append(user_id)

            for user_id in remove_user_ids:
                session_state.live_audio_by_user.pop(user_id, None)

        for user, pcm_bytes in ready:
            try:
                await _process_live_utterance(guild, session_state, vc, user, pcm_bytes)
            except Exception:
                log.exception("Live utterance processing failed user=%s", getattr(user, "id", None))

        await asyncio.sleep(0.25)


def _stop_listening_state(session_state: GuildSession, vc: Optional[discord.VoiceClient]) -> None:
    session_state.listening_enabled = False
    if vc and hasattr(vc, "is_listening") and hasattr(vc, "stop_listening"):
        with contextlib.suppress(Exception):
            if vc.is_listening():
                vc.stop_listening()

    task = session_state.voice_poll_task
    if task and not task.done():
        task.cancel()
    session_state.voice_poll_task = None
    session_state.voice_sink = None
    with session_state.live_audio_lock:
        session_state.live_audio_by_user.clear()


async def generate_ai_reply(
    prompt: str,
    guild: discord.Guild,
    speaker: discord.Member,
    language_code: str,
    conversation_context: str = "",
) -> str:
    if not AI_API_KEY or not AI_MODEL:
        raise RuntimeError("Set AI_API_KEY and AI_MODEL in .env")

    url = f"{AI_API_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json",
    }
    if AI_SITE_URL:
        headers["HTTP-Referer"] = AI_SITE_URL
    if AI_APP_NAME:
        headers["X-Title"] = AI_APP_NAME

    session = await get_http_session()
    preferred_tokens = AI_MAX_TOKENS
    if VOICE_REPLY_MODE == "brief":
        preferred_tokens = max(64, min(AI_MAX_TOKENS, BRIEF_REPLY_MAX_TOKENS))
    candidate_tokens = [preferred_tokens, 1024, 512, 256, 128]
    # Preserve order while removing duplicates and invalid values.
    token_limits = []
    for value in candidate_tokens:
        if value > 0 and value not in token_limits:
            token_limits.append(value)

    text = ""
    last_error = None
    for token_limit in token_limits:
        text = ""
        user_payload = f"Server: {guild.name}\nSpeaker: {speaker.display_name}"
        if conversation_context:
            user_payload += f"\nRecent chat context:\n{conversation_context}"
        user_payload += f"\nCurrent prompt: {prompt}"

        body = {
            "model": AI_MODEL,
            "temperature": AI_TEMPERATURE,
            "max_tokens": token_limit,
            "messages": [
                {
                    "role": "system",
                    "content": build_system_instruction(language_code),
                },
                {
                    "role": "user",
                    "content": user_payload,
                },
            ],
        }

        for attempt in range(1, max(1, AI_RETRY_ATTEMPTS) + 1):
            try:
                async with session.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=AI_REQUEST_TIMEOUT_SECONDS,
                ) as resp:
                    text = await resp.text()
                    if resp.status < 400:
                        break

                    lowered = text.lower()
                    if "max_tokens" in lowered or "token" in lowered or "context" in lowered:
                        last_error = RuntimeError(
                            f"AI request failed ({resp.status}) at max_tokens={token_limit}: {text[:240]}"
                        )
                        break

                    last_error = RuntimeError(f"AI request failed ({resp.status}): {text[:240]}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = RuntimeError(f"Transient AI network failure (attempt {attempt}): {exc}")
                if attempt < max(1, AI_RETRY_ATTEMPTS):
                    await asyncio.sleep(min(1.5 * attempt, 4.0))
                    continue
            else:
                # HTTP call completed without network exception; don't retry this attempt loop.
                pass

            if text:
                # Received response body (possibly error), leave retry loop to evaluate token fallback / final error.
                break

        if text and not last_error:
            break

        # If the current token setting had network/API errors, continue to next token fallback if applicable.
        if last_error and ("max_tokens" in str(last_error).lower() or "token" in str(last_error).lower()):
            continue

        if text:
            raise last_error or RuntimeError("AI request failed.")

    else:
        if last_error:
            raise last_error
        raise RuntimeError("AI request failed after retries.")

    data = json.loads(text)
    try:
        message = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected AI response: {data}") from exc

    if isinstance(message, list):
        chunks = []
        for item in message:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(item.get("text", ""))
        message = "\n".join(chunks)

    return str(message).strip()


async def synthesize_speech(text: str, language_code: str) -> Path:
    output_dir = Path("recordings") / "tts"
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = resolve_tts_backend()
    if backend == "piper" and language_code == "ta":
        # Current bundled piper model is English-only; use edge-tts for Tamil script.
        backend = "edge"
    elif backend == "piper" and language_code == "ta-latn" and not TANGLISH_PREFER_LOCAL_PIPER:
        backend = "edge"

    if backend == "piper":
        output_path = output_dir / f"tts_{int(time.time() * 1000)}.wav"

        def synthesize_local() -> None:
            voice = get_local_tts_voice()
            with wave.open(str(output_path), "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)

        await asyncio.to_thread(synthesize_local)
        return output_path

    if edge_tts is None:
        raise RuntimeError("Install edge-tts to enable spoken replies")

    output_path = output_dir / f"tts_{int(time.time() * 1000)}.mp3"
    voice_name = TTS_VOICE
    if language_code == "ta":
        voice_name = TAMIL_TTS_VOICE
    elif language_code == "ta-latn":
        voice_name = TANGLISH_TTS_VOICE

    await edge_tts.Communicate(text, voice=voice_name, rate=TTS_RATE).save(str(output_path))
    return output_path


async def speak_text(
    vc: discord.VoiceClient,
    session_state: GuildSession,
    text: str,
    language_code: str,
    *,
    sing_mode: bool = False,
) -> None:
    spoken = text.strip()
    if not spoken:
        return

    if VOICE_REPLY_MODE == "brief" and not sing_mode:
        # Faster voice response: speak only the first concise segment while keeping full text in chat.
        first = re.split(r"(?<=[.!?])\s+", spoken, maxsplit=1)[0].strip()
        spoken = first or spoken

    char_limit = min(MAX_TTS_CHARS, VOICE_MAX_CHARS) if VOICE_REPLY_MODE == "brief" else MAX_TTS_CHARS
    if len(spoken) > char_limit:
        spoken = spoken[:char_limit].rsplit(" ", 1)[0].strip() + "..."

    async with session_state.speak_lock:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is not installed on the host, so voice playback cannot start.")

        audio_path = await synthesize_speech(spoken, language_code)
        loop = asyncio.get_running_loop()
        done = loop.create_future()

        def after_playback(error: Optional[Exception]) -> None:
            def resolve() -> None:
                if done.done():
                    return
                if error:
                    done.set_exception(error)
                else:
                    done.set_result(None)

            loop.call_soon_threadsafe(resolve)

        try:
            if vc.is_playing():
                vc.stop()
            vc.play(discord.FFmpegPCMAudio(str(audio_path)), after=after_playback)
            await done
        finally:
            audio_path.unlink(missing_ok=True)


def extract_prompt(message: discord.Message) -> Optional[str]:
    content = message.content.strip()
    if not content:
        return None

    lowered = content.lower()
    triggered = False

    if MENTION_TRIGGER and MENTION_TRIGGER in lowered:
        content = re.sub(re.escape(MENTION_TRIGGER), "", content, flags=re.IGNORECASE)
        triggered = True

    # Support role mention triggers like @fun, which Discord serializes as <@&role_id>.
    if message.role_mentions and any(role.name.lower() == MENTION_KEYWORD for role in message.role_mentions):
        content = re.sub(r"<@&\d+>", "", content)
        triggered = True

    # Allow plain keyword trigger without @ in case Discord strips/sanitizes formatting.
    if not triggered and MENTION_KEYWORD:
        if re.search(rf"(^|\s){re.escape(MENTION_KEYWORD)}(\s|$)", lowered):
            content = re.sub(rf"(^|\s){re.escape(MENTION_KEYWORD)}(\s|$)", " ", content, flags=re.IGNORECASE)
            triggered = True

    if bot.user and bot.user in message.mentions:
        content = content.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "")
        triggered = True

    if not triggered:
        return None

    cleaned = content.strip(" \n\t,:-")
    return cleaned


async def handle_prompt(message: discord.Message, prompt: str) -> None:
    if not message.guild:
        return

    session_state = sessions.get(message.guild.id)
    vc = message.guild.voice_client
    if not session_state or not vc or not vc.is_connected():
        await message.reply("Use !join first so I can answer in voice.")
        return

    if not message.author.voice or message.author.voice.channel != vc.channel:
        await message.reply(f"Join my voice channel: {vc.channel.name}")
        return

    if not prompt:
        await message.reply(f"Send your question after {MENTION_TRIGGER}.")
        return

    language_code = detect_prompt_language(prompt, session_state=session_state, author_id=message.author.id)
    conversation_context = get_recent_chat_context(session_state)

    try:
        response = await generate_ai_reply(
            prompt,
            message.guild,
            message.author,
            language_code,
            conversation_context=conversation_context,
        )
    except Exception as exc:
        log.exception("AI request failed for prompt: %r", prompt)
        response = "I hit a temporary AI connection issue. Please try again in a moment."

    append_chat_history(session_state, message.author.display_name, prompt, response)

    # Always send a reply so the bot never appears unresponsive.
    await message.reply(response)

    try:
        await speak_text(vc, session_state, response, language_code)
    except Exception as exc:
        log.exception("Voice playback failed for prompt: %r", prompt)
        await message.channel.send(f"Text reply sent, but voice playback failed: {exc}")


async def play_song_preview(
    vc: discord.VoiceClient,
    session_state: GuildSession,
    preview_url: str,
) -> None:
    async with session_state.speak_lock:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is not installed on the host, so voice playback cannot start.")

        loop = asyncio.get_running_loop()
        done = loop.create_future()

        def after_playback(error: Optional[Exception]) -> None:
            def resolve() -> None:
                if done.done():
                    return
                if error:
                    done.set_exception(error)
                else:
                    done.set_result(None)

            loop.call_soon_threadsafe(resolve)

        source = discord.FFmpegPCMAudio(
            preview_url,
            before_options="-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5",
        )
        if vc.is_playing():
            vc.stop()
        vc.play(source, after=after_playback)
        await done


async def play_audio_source(
    vc: discord.VoiceClient,
    session_state: GuildSession,
    source_path_or_url: str,
) -> None:
    async with session_state.speak_lock:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is not installed on the host, so voice playback cannot start.")

        loop = asyncio.get_running_loop()
        done = loop.create_future()

        def after_playback(error: Optional[Exception]) -> None:
            def resolve() -> None:
                if done.done():
                    return
                if error:
                    done.set_exception(error)
                else:
                    done.set_result(None)

            loop.call_soon_threadsafe(resolve)

        source = discord.FFmpegPCMAudio(
            source_path_or_url,
            before_options="-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5",
        )
        if vc.is_playing():
            vc.stop()
        vc.play(source, after=after_playback)
        await done


@bot.event
async def on_ready() -> None:
    log.info("Bot ready as %s", bot.user)
    if LIVE_VOICE_ENABLED:
        apply_opus_decode_guard()
    if resolve_tts_backend() == "piper":
        try:
            await asyncio.to_thread(get_local_tts_voice)
            log.info("Local piper voice preloaded")
        except Exception:
            log.exception("Failed to preload local piper voice")
    if LIVE_VOICE_ENABLED and WhisperModel is not None:
        try:
            await asyncio.to_thread(get_asr_model)
            log.info("Local ASR model preloaded")
        except Exception:
            log.exception("Failed to preload local ASR model")


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return

    if message.guild:
        session_state = sessions.get(message.guild.id)
        vc = message.guild.voice_client
        if (
            session_state
            and session_state.pending_lyrics_user_id == message.author.id
            and not message.content.strip().startswith("!")
        ):
            session_state.pending_lyrics_user_id = None
            lyrics = message.content.strip()
            if not lyrics:
                await message.reply("I did not receive any lyrics. Use !sing again.")
                return

            if not vc or not vc.is_connected() or not message.author.voice or message.author.voice.channel != vc.channel:
                await message.reply("Join my voice channel and send !sing again.")
                return

            language_code = detect_prompt_language(
                lyrics,
                session_state=session_state,
                author_id=message.author.id,
            )
            await message.reply("Singing your lyrics now.")
            try:
                await speak_text(vc, session_state, lyrics, language_code, sing_mode=True)
            except Exception as exc:
                log.exception("Lyrics singing failed")
                await message.channel.send(f"Could not sing the lyrics: {exc}")
            return

    if message.guild:
        log.info(
            "Message received guild=%s channel=%s author=%s content=%r mentions=%s role_mentions=%s",
            message.guild.id,
            getattr(message.channel, "name", None),
            message.author,
            message.content,
            [user.id for user in message.mentions],
            [role.name for role in message.role_mentions],
        )

    prompt = extract_prompt(message)
    if prompt is not None:
        log.info("Prompt trigger matched author=%s prompt=%r", message.author, prompt)
        await handle_prompt(message, prompt)

    await bot.process_commands(message)


@bot.command()
async def join(ctx: commands.Context) -> None:
    vc = await ensure_voice_client(ctx)
    if not vc:
        return

    session_state = sessions.get(ctx.guild.id)
    if session_state is None:
        session_state = GuildSession(text_channel_id=ctx.channel.id)
        loaded = _load_chat_memory().get(ctx.guild.id)
        if loaded:
            session_state.chat_history = loaded
        sessions[ctx.guild.id] = session_state
    else:
        session_state.text_channel_id = ctx.channel.id

    if LIVE_VOICE_ENABLED and voice_recv is not None and not isinstance(vc, voice_recv.VoiceRecvClient):
        try:
            await vc.disconnect()
            vc = await ctx.author.voice.channel.connect(timeout=20, cls=voice_recv.VoiceRecvClient)
        except asyncio.TimeoutError:
            await ctx.send("Failed to reconnect with voice receive mode.")
            return

    if LIVE_VOICE_ENABLED and voice_recv is not None:
        apply_opus_decode_guard()

        if not isinstance(vc, voice_recv.VoiceRecvClient):
            await ctx.send("Voice receive mode not available.")
        else:
            def _on_frame(user: Optional[discord.abc.User], data: Any) -> None:
                pcm = getattr(data, "pcm", b"")
                if not isinstance(pcm, (bytes, bytearray)) or not pcm:
                    return
                _push_live_audio_frame(session_state, user, bytes(pcm))

            sink = voice_recv.BasicSink(_on_frame, decode=True)
            vc.listen(sink)

            session_state.listening_enabled = True
            session_state.voice_sink = sink
            if session_state.voice_poll_task and not session_state.voice_poll_task.done():
                session_state.voice_poll_task.cancel()
            session_state.voice_poll_task = asyncio.create_task(_live_voice_poll_loop(ctx.guild.id))
            await ctx.send(
                f"Connected to {vc.channel.name}. Live listening is ON - speak and I will respond! Mention with {MENTION_TRIGGER} for text mode, or use !sing."
            )
            return

    await ctx.send(
        f"Connected to {vc.channel.name}. Mention with {MENTION_TRIGGER} for AI replies, or use !sing / !sing <song name>. Use !listen on for voice chat."
    )


@bot.command()
async def sing(ctx: commands.Context, *, query: str = "") -> None:
    vc = await ensure_voice_client(ctx)
    if not vc:
        return

    session_state = sessions.get(ctx.guild.id)
    if session_state is None:
        session_state = GuildSession(text_channel_id=ctx.channel.id)
        loaded = _load_chat_memory().get(ctx.guild.id)
        if loaded:
            session_state.chat_history = loaded
        sessions[ctx.guild.id] = session_state

    if not query.strip():
        session_state.pending_lyrics_user_id = ctx.author.id
        await ctx.send("Send the lyrics in your next message, and I will sing it.")
        return

    if ENABLE_LOCAL_LIBRARY:
        local_song = search_local_full_song(query)
        if local_song:
            await ctx.send(f"Playing full song from library: {local_song.stem}")
            try:
                await play_audio_source(vc, session_state, str(local_song))
            except Exception as exc:
                log.exception("Local full-song playback failed for query: %r", query)
                await ctx.send(f"Could not play local song right now: {exc}")
            return

    if is_direct_audio_url(query):
        await ctx.send("Playing full audio from the provided URL.")
        try:
            await play_audio_source(vc, session_state, query.strip())
        except Exception as exc:
            log.exception("Direct URL playback failed: %r", query)
            await ctx.send(f"Could not play this URL right now: {exc}")
        return

    try:
        jamendo_track = await search_jamendo_full_track(query)
    except Exception:
        jamendo_track = None

    if jamendo_track:
        await ctx.send(f"Playing full song from Jamendo: {jamendo_track['track']} - {jamendo_track['artist']}")
        try:
            await play_audio_source(vc, session_state, jamendo_track["audio_url"])
        except Exception as exc:
            log.exception("Jamendo full-song playback failed for query: %r", query)
            await ctx.send(f"Could not play Jamendo track right now: {exc}")
        return

    if not ALLOW_PREVIEW_FALLBACK:
        if ENABLE_LOCAL_LIBRARY:
            await ctx.send(
                "No full song found in local library or Jamendo. Add the file to recordings/fullsongs and run !sing <name>. "
                "Use !preview <song name> if you want the 30-second web preview."
            )
        else:
            await ctx.send(
                "No full song found from Jamendo for this query. "
                "Try another title or use !preview <song name> for the 30-second web preview."
            )
        return

    await ctx.send(f"Searching for song: {query}")
    try:
        song = await search_song_preview(query)
        if not song:
            await ctx.send(
                "I could not find a full song in local library and no playable preview online. "
                "Add files to recordings/fullsongs and use !sing <name>, or use !sing and send lyrics manually."
            )
            return

        await ctx.send(f"Playing preview: {song['track']} - {song['artist']}")
        await play_song_preview(vc, session_state, song["preview_url"])
    except Exception as exc:
        log.exception("Song search/playback failed for query: %r", query)
        await ctx.send(f"Could not play that song right now: {exc}")


@bot.command()
async def preview(ctx: commands.Context, *, query: str = "") -> None:
    vc = await ensure_voice_client(ctx)
    if not vc:
        return

    if not query.strip():
        await ctx.send("Usage: !preview <song name>")
        return

    session_state = sessions.get(ctx.guild.id)
    if session_state is None:
        session_state = GuildSession(text_channel_id=ctx.channel.id)
        loaded = _load_chat_memory().get(ctx.guild.id)
        if loaded:
            session_state.chat_history = loaded
        sessions[ctx.guild.id] = session_state

    await ctx.send(f"Searching preview for: {query}")
    try:
        song = await search_song_preview(query)
        if not song:
            await ctx.send("No playable preview found.")
            return

        await ctx.send(f"Playing preview: {song['track']} - {song['artist']}")
        await play_song_preview(vc, session_state, song["preview_url"])
    except Exception as exc:
        log.exception("Preview search/playback failed for query: %r", query)
        await ctx.send(f"Could not play preview right now: {exc}")


@bot.command()
async def listen(ctx: commands.Context, mode: str = "on") -> None:
    session_state = sessions.get(ctx.guild.id)
    if session_state is None:
        session_state = GuildSession(text_channel_id=ctx.channel.id)
        loaded = _load_chat_memory().get(ctx.guild.id)
        if loaded:
            session_state.chat_history = loaded
        sessions[ctx.guild.id] = session_state

    normalized = mode.strip().lower()
    if normalized in {"off", "stop", "0", "false", "no"}:
        _stop_listening_state(session_state, ctx.voice_client)
        await ctx.send("Live listening is off.")
        return

    if not LIVE_VOICE_ENABLED:
        await ctx.send("Live listening is disabled in config. Set LIVE_VOICE_ENABLED=true and restart.")
        return

    apply_opus_decode_guard()

    if voice_recv is None:
        await ctx.send("discord-ext-voice-recv is not installed in this environment.")
        return

    vc = await ensure_voice_client(ctx)
    if not vc:
        return

    if not isinstance(vc, voice_recv.VoiceRecvClient):
        with contextlib.suppress(Exception):
            await vc.disconnect()
        try:
            vc = await ctx.author.voice.channel.connect(timeout=20, cls=voice_recv.VoiceRecvClient)
        except asyncio.TimeoutError:
            await ctx.send("Failed to reconnect with voice receive mode.")
            return

    if vc.is_listening():
        _stop_listening_state(session_state, vc)

    def _on_frame(user: Optional[discord.abc.User], data: Any) -> None:
        pcm = getattr(data, "pcm", b"")
        if not isinstance(pcm, (bytes, bytearray)) or not pcm:
            return
        _push_live_audio_frame(session_state, user, bytes(pcm))

    sink = voice_recv.BasicSink(_on_frame, decode=True)
    vc.listen(sink)

    session_state.listening_enabled = True
    session_state.voice_sink = sink
    if session_state.voice_poll_task and not session_state.voice_poll_task.done():
        session_state.voice_poll_task.cancel()
    session_state.voice_poll_task = asyncio.create_task(_live_voice_poll_loop(ctx.guild.id))

    await ctx.send("Live listening enabled. Speak in voice and I will reply.")


@bot.command()
async def leave(ctx: commands.Context) -> None:
    vc = ctx.voice_client

    if not vc:
        await ctx.send("I am not in a voice channel.")
        return

    if vc.is_playing():
        vc.stop()

    session_state = sessions.get(ctx.guild.id)
    if session_state:
        _stop_listening_state(session_state, vc)

    await vc.disconnect()
    await ctx.send("Disconnected.")


async def main() -> None:
    global _discord_login_attempts, _discord_last_error

    token = os.getenv("DISCORD_BOT_TOKEN", "")
    if not token:
        raise RuntimeError("Set DISCORD_BOT_TOKEN in .env")

    runner = await start_healthcheck_server()
    try:
        attempts = 0
        while True:
            attempts += 1
            _discord_login_attempts = attempts
            try:
                await bot.start(token)
                _discord_last_error = None
                break
            except discord.LoginFailure as exc:
                _discord_last_error = "Discord login failed. Check DISCORD_BOT_TOKEN value."
                raise RuntimeError(_discord_last_error) from exc
            except discord.HTTPException as exc:
                status = getattr(exc, "status", None)
                text = str(exc).lower()
                is_rate_limited = status == 429 or "error 1015" in text or "rate limited" in text
                if not is_rate_limited:
                    _discord_last_error = f"Discord HTTPException status={status}: {exc}"
                    raise

                if attempts > max(1, DISCORD_LOGIN_MAX_RETRIES):
                    raise RuntimeError(
                        "Discord login kept failing due to rate limiting. "
                        "Wait for the Cloudflare ban window to expire or redeploy to get a new egress IP."
                    ) from exc

                delay = min(
                    DISCORD_LOGIN_RETRY_BASE_SECONDS * (2 ** (attempts - 1)),
                    DISCORD_LOGIN_RETRY_MAX_SECONDS,
                )
                # Add jitter so multiple restarts do not hit Discord simultaneously.
                delay += random.uniform(0, min(5.0, delay * 0.15))
                _discord_last_error = (
                    f"Rate limited by Discord/Cloudflare (status={status}). Retrying in {delay:.1f}s"
                )
                log.warning(
                    "Discord login rate-limited (attempt %s/%s). Retrying in %.1fs",
                    attempts,
                    max(1, DISCORD_LOGIN_MAX_RETRIES),
                    delay,
                )
                await asyncio.sleep(delay)
    finally:
        if _http_session and not _http_session.closed:
            await _http_session.close()
        await runner.cleanup()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
