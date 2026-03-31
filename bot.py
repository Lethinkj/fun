from __future__ import annotations

import asyncio
import audioop
import contextlib
import datetime as dt
import json
import logging
import os
import re
import tempfile
import threading
import time
import wave
from collections import defaultdict, deque
from dataclasses import dataclass, field
from difflib import get_close_matches
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

import aiohttp
import discord
from discord import opus as discord_opus
from discord.ext import commands, voice_recv
from dotenv import load_dotenv
from aiohttp import web

try:
    import edge_tts
except ImportError:  # pragma: no cover - handled at runtime
    edge_tts = None

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - handled at runtime
    WhisperModel = None

try:
    from piper import PiperVoice
except ImportError:  # pragma: no cover - handled at runtime
    PiperVoice = None

load_dotenv()

HF_ASR_MODEL = os.getenv("HF_ASR_MODEL", "openai/whisper-large-v3-turbo")
HF_SUMMARY_MODEL = os.getenv("HF_SUMMARY_MODEL", "philschmid/bart-large-cnn-samsum")
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
RECORDINGS_DIR = Path(os.getenv("RECORDINGS_DIR", "recordings"))
WAKE_WORDS = tuple(
    word.strip().lower()
    for word in os.getenv("WAKE_WORDS", "hey bee,hey b,hey bot").split(",")
    if word.strip()
)
VOICE_CHUNK_SECONDS = float(os.getenv("VOICE_CHUNK_SECONDS", "4"))
VOICE_POLL_SECONDS = float(os.getenv("VOICE_POLL_SECONDS", "1.5"))
AI_API_BASE_URL = os.getenv("AI_API_BASE_URL", "https://openrouter.ai/api/v1")
AI_API_KEY = os.getenv("AI_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))
AI_MODEL = os.getenv("AI_MODEL", os.getenv("OPENROUTER_MODEL", ""))
AI_SITE_URL = os.getenv("AI_SITE_URL", "")
AI_APP_NAME = os.getenv("AI_APP_NAME", "Bee Voice Bot")
AI_TEMPERATURE = float(os.getenv("AI_TEMPERATURE", "0.5"))
AI_MAX_TOKENS = int(os.getenv("AI_MAX_TOKENS", "220"))
TTS_BACKEND = os.getenv("TTS_BACKEND", "piper").lower()
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AriaNeural")
MAX_TTS_CHARS = int(os.getenv("MAX_TTS_CHARS", "450"))
LOCAL_TTS_MODEL_PATH = Path(os.getenv("LOCAL_TTS_MODEL_PATH", "voices/en_US-lessac-medium.onnx"))
MIN_RMS_FOR_TRANSCRIPTION = int(os.getenv("MIN_RMS_FOR_TRANSCRIPTION", "140"))
ASR_BACKEND = os.getenv("ASR_BACKEND", "auto").lower()
LOCAL_ASR_MODEL = os.getenv("LOCAL_ASR_MODEL", "tiny.en")
LOCAL_ASR_DEVICE = os.getenv("LOCAL_ASR_DEVICE", "cpu")
LOCAL_ASR_COMPUTE_TYPE = os.getenv("LOCAL_ASR_COMPUTE_TYPE", "int8")
LIVE_VOICE_ENABLED = os.getenv("LIVE_VOICE_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("PORT", os.getenv("WEB_PORT", "10000")))

PCM_BYTES_PER_SECOND = 48000 * 2 * 2
MIN_CHUNK_BYTES = int(VOICE_CHUNK_SECONDS * PCM_BYTES_PER_SECOND)
SILENT_PCM_FRAME = b"\x00" * discord_opus.Decoder.FRAME_SIZE

log = logging.getLogger(__name__)
_decode_warning_times: dict[int, float] = {}
_local_whisper_model: Optional[Any] = None
_local_whisper_lock = threading.Lock()
_local_tts_voice: Optional[Any] = None
_local_tts_lock = threading.Lock()

def configure_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if any(isinstance(handler, RotatingFileHandler) and getattr(handler, "baseFilename", "").endswith("bot.log") for handler in root.handlers):
        return

    file_handler = RotatingFileHandler("bot.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(file_handler)


configure_logging()


class LiveVoiceSink(voice_recv.AudioSink):
    """Stores received PCM audio and speaker metadata for rolling transcription."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._base_offset = 0
        self._buffer = bytearray()
        self._events: deque[tuple[int, int, int]] = deque()

    def wants_opus(self) -> bool:
        return False

    def write(self, user: Optional[discord.User], data: voice_recv.VoiceData) -> None:
        if not data.pcm:
            return

        user_id = getattr(user, "id", 0)
        with self._lock:
            self._buffer.extend(data.pcm)
            absolute_end = self._base_offset + len(self._buffer)
            self._events.append((absolute_end, user_id, len(data.pcm)))

    def cleanup(self) -> None:
        pass

    def byte_length(self) -> int:
        with self._lock:
            return self._base_offset + len(self._buffer)

    def slice_bytes(self, start: int, end: Optional[int] = None) -> bytes:
        with self._lock:
            local_start = max(0, start - self._base_offset)
            local_end = None if end is None else max(0, end - self._base_offset)
            return bytes(self._buffer[local_start:local_end])

    def trim_before(self, offset: int) -> None:
        with self._lock:
            if offset <= self._base_offset:
                return

            trim_length = min(offset - self._base_offset, len(self._buffer))
            if trim_length <= 0:
                return

            del self._buffer[:trim_length]
            self._base_offset += trim_length

            while self._events and self._events[0][0] <= self._base_offset:
                self._events.popleft()

    def dominant_speaker_id(self, start: int, end: int) -> Optional[int]:
        totals: dict[int, int] = defaultdict(int)
        with self._lock:
            for event_end, user_id, size in self._events:
                if event_end <= start:
                    continue
                if event_end > end:
                    break
                if user_id:
                    totals[user_id] += size

        if not totals:
            return None
        return max(totals.items(), key=lambda item: item[1])[0]


@dataclass
class GuildSession:
    sink: LiveVoiceSink
    started_at: dt.datetime
    text_channel_id: int
    live_task: Optional[asyncio.Task] = None
    manual_capture_offset: Optional[int] = None
    last_processed_offset: int = 0
    speak_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    processing_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_trigger_at: float = 0.0
    last_asr_warning_at: float = 0.0


sessions: dict[int, GuildSession] = {}

intents = discord.Intents.default()
intents.guilds = True
intents.message_content = True
intents.voice_states = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)


def patch_voice_recv_decoder() -> None:
    """Keep the voice receive thread alive and ignore packets from unmapped speakers."""
    packet_decoder_cls = voice_recv.opus.PacketDecoder
    original_decode_packet = packet_decoder_cls._decode_packet
    original_process_packet = packet_decoder_cls._process_packet

    if getattr(packet_decoder_cls, "_bee_patched_decoder", False):
        return

    def safe_decode_packet(self, packet):
        try:
            return original_decode_packet(self, packet)
        except discord_opus.OpusError as exc:
            now = time.monotonic()
            last_logged = _decode_warning_times.get(self.ssrc, 0.0)
            if now - last_logged >= 5:
                _decode_warning_times[self.ssrc] = now
                log.warning("Recovered from corrupted opus packet on ssrc=%s: %s", self.ssrc, exc)
            self._decoder = None if self.sink.wants_opus() else discord_opus.Decoder()
            return packet, SILENT_PCM_FRAME

    def safe_process_packet(self, packet):
        member = self._get_cached_member()
        if member is None:
            self._cached_id = self.sink.voice_client._get_id_from_ssrc(self.ssrc)  # type: ignore[attr-defined]
            member = self._get_cached_member()

        if member is None:
            self._last_seq = packet.sequence
            self._last_ts = packet.timestamp
            return voice_recv.opus.VoiceData(packet, None, pcm=b"")

        return original_process_packet(self, packet)

    packet_decoder_cls._decode_packet = safe_decode_packet
    packet_decoder_cls._process_packet = safe_process_packet
    packet_decoder_cls._bee_patched_decoder = True


patch_voice_recv_decoder()

logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.WARNING)
logging.getLogger("discord.ext.voice_recv.gateway").setLevel(logging.WARNING)
logging.getLogger("discord.ext.voice_recv.opus").setLevel(logging.WARNING)


async def health_handler(_request: web.Request) -> web.Response:
    return web.json_response(
        {
            "status": "ok",
            "bot_ready": bot.is_ready(),
            "bot_user": str(bot.user) if bot.user else None,
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


@bot.event
async def on_ready() -> None:
    log.info("Bot ready as %s (id=%s)", bot.user, getattr(bot.user, "id", None))
    log.info("AI configured=%s model=%s tts_backend=%s", bool(AI_API_KEY and AI_MODEL), AI_MODEL or None, TTS_BACKEND)
    print(f"Logged in as {bot.user}")


async def ensure_voice_client(ctx: commands.Context) -> Optional[voice_recv.VoiceRecvClient]:
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("You need to be in a voice channel for me to join.")
        return None

    permissions = ctx.author.voice.channel.permissions_for(ctx.guild.me)
    if not permissions.connect or not permissions.speak:
        await ctx.send("I need the 'Connect' and 'Speak' permissions to join the voice channel.")
        return None

    channel = ctx.author.voice.channel
    current = ctx.voice_client

    if current and current.channel.id != channel.id:
        await current.move_to(channel)

    if not ctx.voice_client:
        try:
            vc = await channel.connect(cls=voice_recv.VoiceRecvClient, timeout=60)
        except asyncio.TimeoutError:
            await ctx.send("Failed to connect to the voice channel: Connection timed out.")
            return None
    else:
        vc = ctx.voice_client

    if not isinstance(vc, voice_recv.VoiceRecvClient):
        await ctx.send("Reconnect me with `!leave` then `!join` so I can receive voice.")
        return None

    return vc


def write_wav(path: Path, pcm_data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(48000)
        wav_file.writeframes(pcm_data)


def trim_processed_audio(session_state: GuildSession) -> None:
    trim_offset = (
        session_state.manual_capture_offset
        if session_state.manual_capture_offset is not None
        else session_state.last_processed_offset
    )
    session_state.sink.trim_before(trim_offset)


def should_transcribe_pcm(pcm_data: bytes) -> bool:
    if not pcm_data:
        return False

    try:
        return audioop.rms(pcm_data, 2) >= MIN_RMS_FOR_TRANSCRIPTION
    except audioop.error:
        return True


async def read_json_response(resp: aiohttp.ClientResponse) -> tuple[str, object]:
    raw = await resp.read()
    text = raw.decode("utf-8", errors="replace")
    if not text.strip():
        return text, {}

    try:
        return text, json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response: {text[:200]}") from exc


async def transcribe_pcm_bytes(session: aiohttp.ClientSession, pcm_data: bytes) -> str:
    if not HF_API_TOKEN:
        raise RuntimeError("Missing HUGGINGFACE_API_TOKEN in .env")

    if not pcm_data:
        return ""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        write_wav(temp_path, pcm_data)
        return await transcribe_audio_path(session, temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def get_local_whisper_model() -> Any:
    if WhisperModel is None:
        raise RuntimeError("Local ASR is not installed. Install faster-whisper first.")

    global _local_whisper_model
    with _local_whisper_lock:
        if _local_whisper_model is None:
            log.info(
                "Loading local ASR model %s on %s (%s)",
                LOCAL_ASR_MODEL,
                LOCAL_ASR_DEVICE,
                LOCAL_ASR_COMPUTE_TYPE,
            )
            _local_whisper_model = WhisperModel(
                LOCAL_ASR_MODEL,
                device=LOCAL_ASR_DEVICE,
                compute_type=LOCAL_ASR_COMPUTE_TYPE,
            )
        return _local_whisper_model


def get_local_tts_voice() -> Any:
    if PiperVoice is None:
        raise RuntimeError("Local TTS is not installed. Install piper-tts first.")

    global _local_tts_voice
    with _local_tts_lock:
        if _local_tts_voice is None:
            if not LOCAL_TTS_MODEL_PATH.exists():
                raise RuntimeError(f"Local TTS model not found: {LOCAL_TTS_MODEL_PATH}")
            log.info("Loading local TTS model from %s", LOCAL_TTS_MODEL_PATH)
            _local_tts_voice = PiperVoice.load(LOCAL_TTS_MODEL_PATH)
        return _local_tts_voice


def transcribe_audio_path_local(audio_path: Path) -> str:
    model = get_local_whisper_model()
    segments, _info = model.transcribe(
        str(audio_path),
        beam_size=1,
        best_of=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text


async def transcribe_audio_path(session: aiohttp.ClientSession, audio_path: Path) -> str:
    use_local_first = ASR_BACKEND == "local"
    use_remote = ASR_BACKEND in {"auto", "remote"}
    use_local_fallback = ASR_BACKEND in {"auto", "local"}

    if use_local_first:
        return await asyncio.to_thread(transcribe_audio_path_local, audio_path)

    remote_error: Optional[Exception] = None
    if use_remote:
        try:
            return await hf_inference_audio(session, audio_path)
        except Exception as exc:
            remote_error = exc
            log.warning("Remote ASR failed, trying local fallback: %s", exc)

    if use_local_fallback:
        try:
            return await asyncio.to_thread(transcribe_audio_path_local, audio_path)
        except Exception as local_exc:
            if remote_error is not None:
                raise RuntimeError(f"Remote ASR failed: {remote_error}; local ASR failed: {local_exc}") from local_exc
            raise

    if remote_error is not None:
        raise remote_error
    raise RuntimeError("No ASR backend is enabled.")


async def hf_inference_audio(session: aiohttp.ClientSession, audio_path: Path) -> str:
    url = f"https://api-inference.huggingface.co/models/{HF_ASR_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    payload = audio_path.read_bytes()
    for _ in range(3):
        try:
            async with session.post(url, headers=headers, data=payload, timeout=240) as resp:
                text, data = await read_json_response(resp)
                if resp.status == 503:
                    await asyncio.sleep(5)
                    continue
                if resp.status >= 400:
                    raise RuntimeError(f"ASR failed ({resp.status}): {text[:200]}")
                if isinstance(data, dict) and "text" in data:
                    return data["text"].strip()
                raise RuntimeError(f"Unexpected ASR response: {data}")
        except (aiohttp.ClientPayloadError, aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
            log.warning("Transient ASR request failure: %s", exc)
            await asyncio.sleep(2)

    raise RuntimeError("ASR model is still warming up.")


async def hf_inference_summary(session: aiohttp.ClientSession, transcript: str) -> str:
    url = f"https://api-inference.huggingface.co/models/{HF_SUMMARY_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }

    chunk = transcript[:3500]
    body = {"inputs": chunk}

    for _ in range(3):
        try:
            async with session.post(url, headers=headers, json=body, timeout=180) as resp:
                text, data = await read_json_response(resp)
                if resp.status == 503:
                    await asyncio.sleep(5)
                    continue
                if resp.status >= 400:
                    raise RuntimeError(f"Summary failed ({resp.status}): {text[:200]}")
                if isinstance(data, list) and data and "summary_text" in data[0]:
                    return data[0]["summary_text"].strip()
                if isinstance(data, dict) and "summary_text" in data:
                    return data["summary_text"].strip()
                raise RuntimeError(f"Unexpected summary response: {data}")
        except (aiohttp.ClientPayloadError, aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
            log.warning("Transient summary request failure: %s", exc)
            await asyncio.sleep(2)

    raise RuntimeError("Summary model was still loading. Try again in about a minute.")


async def generate_ai_reply(prompt: str, guild: discord.Guild, speaker: Optional[discord.Member]) -> str:
    if not AI_API_KEY or not AI_MODEL:
        raise RuntimeError("Set AI_API_KEY and AI_MODEL in .env to enable AI replies.")

    url = f"{AI_API_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json",
    }
    if AI_SITE_URL:
        headers["HTTP-Referer"] = AI_SITE_URL
    if AI_APP_NAME:
        headers["X-Title"] = AI_APP_NAME

    speaker_name = speaker.display_name if speaker else "a user"
    body = {
        "model": AI_MODEL,
        "temperature": AI_TEMPERATURE,
        "max_tokens": AI_MAX_TOKENS,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Bee, a Discord voice assistant. "
                    "Reply with short, clear, spoken-friendly answers. "
                    "Keep most responses under 3 sentences unless the user asks for detail."
                ),
            },
            {
                "role": "user",
                "content": f"Server: {guild.name}\nSpeaker: {speaker_name}\nPrompt: {prompt}",
            },
        ],
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=body, timeout=120) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"AI request failed ({resp.status}): {text[:240]}")
            data = await resp.json()

    try:
        message = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected AI response: {data}") from exc

    if isinstance(message, list):
        parts = []
        for item in message:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        message = "\n".join(parts)

    return str(message).strip()


def extract_wake_command(transcript: str) -> Optional[str]:
    lowered = transcript.lower()
    for wake_word in WAKE_WORDS:
        idx = lowered.find(wake_word)
        if idx == -1:
            continue
        command = transcript[idx + len(wake_word):].strip(" ,.!?")
        return command or ""
    return None


def parse_moderation_command(command: str) -> Optional[tuple[str, str]]:
    lowered = command.lower().strip()
    patterns = (
        ("unmute ", "unmute"),
        ("mute ", "mute"),
        ("undeafen ", "undeafen"),
        ("deafen ", "deafen"),
        ("kick ", "kick"),
        ("disconnect ", "disconnect"),
    )
    for prefix, action in patterns:
        if lowered.startswith(prefix):
            return action, command[len(prefix):].strip()

    action_words = {
        "mute": "mute",
        "unmute": "unmute",
        "deafen": "deafen",
        "undeafen": "undeafen",
        "kick": "kick",
        "disconnect": "disconnect",
    }
    for word, action in action_words.items():
        marker = f"{word} "
        if marker in lowered:
            idx = lowered.find(marker)
            return action, command[idx + len(marker):].strip()
    return None


def find_member_by_name(guild: discord.Guild, raw_name: str) -> Optional[discord.Member]:
    target = raw_name.strip().lower()
    if not target:
        return None

    members = list(guild.members)
    exact = [
        member
        for member in members
        if target in {member.display_name.lower(), member.name.lower(), str(member).lower()}
    ]
    if exact:
        return exact[0]

    substring = [
        member
        for member in members
        if target in member.display_name.lower()
        or target in member.name.lower()
        or target in str(member).lower()
    ]
    if substring:
        return substring[0]

    lookup = {
        member.display_name.lower(): member for member in members
    }
    lookup.update({member.name.lower(): member for member in members})
    lookup.update({str(member).lower(): member for member in members})

    match = get_close_matches(target, list(lookup.keys()), n=1, cutoff=0.6)
    return lookup[match[0]] if match else None


def get_session_text_channel(guild: discord.Guild) -> Optional[discord.abc.Messageable]:
    session_state = sessions.get(guild.id)
    if not session_state:
        return None
    return bot.get_channel(session_state.text_channel_id)


async def send_status(guild: discord.Guild, message: str) -> None:
    channel = get_session_text_channel(guild)
    if channel is not None:
        await channel.send(message)


async def synthesize_speech(text: str) -> Path:
    output_dir = RECORDINGS_DIR / "tts"
    output_dir.mkdir(parents=True, exist_ok=True)
    if TTS_BACKEND == "piper":
        output_path = output_dir / f"tts_{int(time.time() * 1000)}.wav"

        def synthesize_local() -> None:
            voice = get_local_tts_voice()
            with wave.open(str(output_path), "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)

        await asyncio.to_thread(synthesize_local)
        return output_path

    if edge_tts is None:
        raise RuntimeError("Install edge-tts to enable spoken replies.")

    output_path = output_dir / f"tts_{int(time.time() * 1000)}.mp3"
    communicate = edge_tts.Communicate(text, voice=TTS_VOICE)
    await communicate.save(str(output_path))
    return output_path


async def speak_text(vc: voice_recv.VoiceRecvClient, session_state: GuildSession, text: str) -> None:
    spoken = text.strip()
    if not spoken:
        return

    if len(spoken) > MAX_TTS_CHARS:
        spoken = spoken[:MAX_TTS_CHARS].rsplit(" ", 1)[0].strip() + "..."

    async with session_state.speak_lock:
        log.info(
            "Starting TTS playback in guild voice channel=%s chars=%s backend=%s",
            getattr(vc.channel, "name", None),
            len(spoken),
            TTS_BACKEND,
        )
        audio_path = await synthesize_speech(spoken)
        loop = asyncio.get_running_loop()
        finished = loop.create_future()

        def after_playback(error: Optional[Exception]) -> None:
            def resolve() -> None:
                if finished.done():
                    return
                if error:
                    finished.set_exception(error)
                else:
                    finished.set_result(None)

            loop.call_soon_threadsafe(resolve)

        try:
            if vc.is_playing():
                log.info("Voice client already playing; stopping previous audio first")
                vc.stop()
            source = discord.FFmpegPCMAudio(str(audio_path))
            log.info("Starting FFmpeg playback from %s", audio_path)
            vc.play(source, after=after_playback)
            await finished
        finally:
            audio_path.unlink(missing_ok=True)


async def execute_moderation_action(
    guild: discord.Guild,
    speaker: Optional[discord.Member],
    action: str,
    target_name: str,
) -> str:
    if speaker is None:
        return "I could not figure out who asked, so I did not change anyone's voice permissions."

    target = find_member_by_name(guild, target_name)
    if target is None:
        return f"I couldn't find anyone matching {target_name!r}."

    if target == guild.me:
        return "I won't moderate myself."

    if action == "kick":
        if not speaker.guild_permissions.kick_members:
            return "You do not have permission to kick members."
        await target.kick(reason=f"Voice command by {speaker}")
        return f"Kicked {target.display_name}."

    if action in {"mute", "unmute", "deafen", "undeafen"}:
        permission_name = "mute_members" if action in {"mute", "unmute"} else "deafen_members"
        if not getattr(speaker.guild_permissions, permission_name):
            return "You do not have permission for that voice moderation action."
        if not target.voice:
            return f"{target.display_name} is not in a voice channel."

        kwargs = {}
        if action == "mute":
            kwargs["mute"] = True
        elif action == "unmute":
            kwargs["mute"] = False
        elif action == "deafen":
            kwargs["deafen"] = True
        elif action == "undeafen":
            kwargs["deafen"] = False

        await target.edit(reason=f"Voice command by {speaker}", **kwargs)
        verb = {
            "mute": "Muted",
            "unmute": "Unmuted",
            "deafen": "Deafened",
            "undeafen": "Undeafened",
        }[action]
        return f"{verb} {target.display_name}."

    if action == "disconnect":
        if not speaker.guild_permissions.move_members:
            return "You do not have permission to disconnect members from voice."
        if not target.voice:
            return f"{target.display_name} is not in a voice channel."
        await target.move_to(None, reason=f"Voice command by {speaker}")
        return f"Disconnected {target.display_name} from voice."

    return "I heard a moderation request, but I do not support that action yet."


async def respond_to_voice_command(
    guild: discord.Guild,
    vc: voice_recv.VoiceRecvClient,
    session_state: GuildSession,
    command_text: str,
    speaker: Optional[discord.Member],
) -> None:
    if not command_text:
        response = "Yes? I'm listening."
    else:
        moderation = parse_moderation_command(command_text)
        if moderation:
            response = await execute_moderation_action(guild, speaker, moderation[0], moderation[1])
        else:
            response = await generate_ai_reply(command_text, guild, speaker)

    speaker_label = speaker.display_name if speaker else "Unknown speaker"
    await send_status(guild, f"**{speaker_label}** said: `{command_text or '(wake word only)'}`\n**Bee:** {response}")
    await speak_text(vc, session_state, response)


async def respond_to_text_prompt(message: discord.Message, prompt: str) -> None:
    log.info("Text prompt received in guild=%s from user=%s: %s", getattr(message.guild, "id", None), message.author, prompt)

    if not message.guild:
        await message.reply("Use this in a server where I can join a voice channel.")
        return

    session_state = sessions.get(message.guild.id)
    vc = message.guild.voice_client
    log.info(
        "Text prompt state guild=%s has_session=%s voice_client=%s connected=%s user_voice=%s",
        message.guild.id,
        bool(session_state),
        type(vc).__name__ if vc else None,
        vc.is_connected() if vc else False,
        getattr(getattr(message.author, "voice", None), "channel", None),
    )
    if not session_state or not isinstance(vc, voice_recv.VoiceRecvClient) or not vc.is_connected():
        await message.reply("Join me to a voice channel with `!join` first, then mention me.")
        return

    if message.author.voice and vc.channel != message.author.voice.channel:
        await message.reply(f"I am in **{vc.channel.name}**. Join that voice channel or move me there with `!join`.")
        return

    async with session_state.processing_lock:
        try:
            ai_started = time.monotonic()
            response = await generate_ai_reply(prompt, message.guild, message.author)
            log.info("AI response ready in %.2fs for user=%s", time.monotonic() - ai_started, message.author)
        except Exception as exc:
            log.exception("AI request failed for mention prompt")
            await message.reply(f"AI request failed: `{exc}`")
            return

        await message.reply(response)

        try:
            tts_started = time.monotonic()
            await speak_text(vc, session_state, response)
            log.info("Voice playback completed in %.2fs for user=%s", time.monotonic() - tts_started, message.author)
        except Exception as exc:
            log.exception("Voice playback failed for mention prompt")
            await message.channel.send(f"I answered in text, but voice playback failed: `{exc}`")


async def monitor_live_session(guild: discord.Guild, vc: voice_recv.VoiceRecvClient, session_state: GuildSession) -> None:
    async with aiohttp.ClientSession() as http_session:
        while True:
            await asyncio.sleep(VOICE_POLL_SECONDS)

            if guild.id not in sessions:
                return
            if not vc.is_connected() or vc.is_playing():
                continue

            total_bytes = session_state.sink.byte_length()
            pending_bytes = total_bytes - session_state.last_processed_offset
            if pending_bytes < MIN_CHUNK_BYTES:
                continue

            start = session_state.last_processed_offset
            end = total_bytes
            session_state.last_processed_offset = end
            pcm_data = session_state.sink.slice_bytes(start, end)
            if not pcm_data:
                continue
            if not should_transcribe_pcm(pcm_data):
                trim_processed_audio(session_state)
                continue

            try:
                transcript = await transcribe_pcm_bytes(http_session, pcm_data)
            except Exception as exc:
                message = str(exc)
                if "warming up" in message.lower():
                    now = time.monotonic()
                    if now - session_state.last_asr_warning_at >= 60:
                        session_state.last_asr_warning_at = now
                        await send_status(guild, "Speech recognition is warming up. I’ll keep retrying in the background.")
                    trim_processed_audio(session_state)
                    await asyncio.sleep(8)
                    continue

                await send_status(guild, f"Live transcription failed: `{exc}`")
                trim_processed_audio(session_state)
                await asyncio.sleep(3)
                continue

            if not transcript.strip():
                trim_processed_audio(session_state)
                continue

            command_text = extract_wake_command(transcript)
            if command_text is not None:
                now = time.monotonic()
                if now - session_state.last_trigger_at >= 4:
                    speaker_id = session_state.sink.dominant_speaker_id(start, end)
                    speaker = guild.get_member(speaker_id) if speaker_id else None
                    session_state.last_trigger_at = now

                    async with session_state.processing_lock:
                        try:
                            await respond_to_voice_command(guild, vc, session_state, command_text, speaker)
                        except Exception as exc:
                            await send_status(guild, f"Voice command failed: `{exc}`")

            trim_processed_audio(session_state)


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot or not bot.user:
        return

    if message.guild:
        interesting = "bee" in message.content.lower() or str(bot.user.id) in message.content or bool(message.mentions)
        if interesting:
            log.info(
                "Incoming message guild=%s channel=%s author=%s content=%r mentions=%s",
                message.guild.id,
                getattr(message.channel, "name", None),
                message.author,
                message.content,
                [user.id for user in message.mentions],
            )

    lowered = message.content.lower()
    names = {
        bot.user.name.lower(),
        bot.user.display_name.lower(),
        "beelert",
        "bee lert",
    }
    mentioned = bot.user in message.mentions or any(
        token in lowered
        for token in (
            f"<@{bot.user.id}>",
            f"<@!{bot.user.id}>",
            *(f"@{name}" for name in names),
        )
    )

    if mentioned:
        cleaned = message.content
        cleaned = cleaned.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "")
        for name in names:
            cleaned = re.sub(rf"@?{re.escape(name)}", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip(" \n\t,:-")
        log.info("Mention detected from user=%s raw=%r cleaned=%r mentions=%s", message.author, message.content, cleaned, [u.id for u in message.mentions])
        if cleaned:
            await respond_to_text_prompt(message, cleaned)
        else:
            await message.reply("Ask me something after mentioning me, and I'll answer in voice.")

    await bot.process_commands(message)


@bot.command()
async def join(ctx: commands.Context) -> None:
    vc = await ensure_voice_client(ctx)
    if not vc:
        return

    log.info(
        "Join command guild=%s author=%s channel=%s live_voice=%s",
        ctx.guild.id if ctx.guild else None,
        ctx.author,
        getattr(vc.channel, "name", None),
        LIVE_VOICE_ENABLED,
    )

    existing = sessions.get(ctx.guild.id)
    if existing:
        existing.text_channel_id = ctx.channel.id
        mode = "live listening" if existing.live_task else "mention mode"
        await ctx.send(f"Already active in **{vc.channel.name}** with {mode}. Use `!leave` when you want me to stop.")
        return

    sink = LiveVoiceSink()
    session_state = GuildSession(sink=sink, started_at=dt.datetime.now(dt.timezone.utc), text_channel_id=ctx.channel.id)
    sessions[ctx.guild.id] = session_state
    if LIVE_VOICE_ENABLED:
        vc.listen(sink)
        session_state.live_task = asyncio.create_task(monitor_live_session(ctx.guild, vc, session_state))
        await ctx.send(
            f"Connected to **{vc.channel.name}** and listening live. "
            f"Say `{WAKE_WORDS[0]}` followed by your request, or mention me in text. Use `!leave` to stop."
        )
    else:
        await ctx.send(
            f"Connected to **{vc.channel.name}**. "
            f"Mention me in text, like `@{bot.user.display_name} what is recursion`, and I'll speak back here."
        )


@bot.command()
async def startlisten(ctx: commands.Context) -> None:
    session_state = sessions.get(ctx.guild.id)
    if not session_state:
        await ctx.send("Use `!join` first so I can enter voice and listen.")
        return

    session_state.manual_capture_offset = session_state.sink.byte_length()
    await ctx.send("Manual capture started. Use `!stoplisten` for a transcript and summary of what happened next.")


@bot.command()
async def stoplisten(ctx: commands.Context) -> None:
    session_state = sessions.get(ctx.guild.id)
    if not session_state or session_state.manual_capture_offset is None:
        await ctx.send("No manual capture is running. Start one with `!startlisten`.")
        return

    start = session_state.manual_capture_offset
    session_state.manual_capture_offset = None
    end = session_state.sink.byte_length()
    pcm_data = session_state.sink.slice_bytes(start, end)

    if not pcm_data:
        await ctx.send("No voice captured. Speak for a few seconds and try again.")
        return

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = RECORDINGS_DIR / f"{ctx.guild.id}_{ts}.wav"
    write_wav(wav_path, pcm_data)
    await ctx.send("Processing captured audio...")

    try:
        async with aiohttp.ClientSession() as http_session:
            transcript = await hf_inference_audio(http_session, wav_path)
            summary = await hf_inference_summary(http_session, transcript)
    except Exception as exc:
        await ctx.send(f"Failed to process audio: `{exc}`")
        return

    transcript_preview = transcript[:1700] + ("..." if len(transcript) > 1700 else "")
    await ctx.send(
        f"**Summary**\n{summary}\n\n"
        f"**Transcript (preview)**\n{transcript_preview}\n\n"
        f"Saved recording: `{wav_path}`"
    )


@bot.command()
async def leave(ctx: commands.Context) -> None:
    vc = ctx.voice_client
    session_state = sessions.pop(ctx.guild.id, None)

    if session_state and session_state.live_task:
        session_state.live_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await session_state.live_task
    elif session_state:
        await ctx.send("No active task to cancel.")

    if not vc:
        await ctx.send("I am not in a voice channel.")
        return

    if isinstance(vc, voice_recv.VoiceRecvClient) and vc.is_listening():
        vc.stop_listening()
    if vc.is_playing():
        vc.stop()

    await vc.disconnect()
    await ctx.send("Disconnected.")


async def main() -> None:
    token = os.getenv("DISCORD_BOT_TOKEN", "")
    if not token:
        raise RuntimeError("Set DISCORD_BOT_TOKEN in .env")

    runner = await start_healthcheck_server()
    try:
        await bot.start(token)
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
