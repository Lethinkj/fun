from __future__ import annotations

import asyncio
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

TTS_BACKEND = os.getenv("TTS_BACKEND", "auto").lower()
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AriaNeural")
TTS_RATE = os.getenv("TTS_RATE", "+20%")
MAX_TTS_CHARS = int(os.getenv("MAX_TTS_CHARS", "2000"))
VOICE_REPLY_MODE = os.getenv("VOICE_REPLY_MODE", "brief").lower()
VOICE_MAX_CHARS = int(os.getenv("VOICE_MAX_CHARS", "260"))
LOCAL_TTS_MODEL_PATH = Path(os.getenv("LOCAL_TTS_MODEL_PATH", "voices/en_US-lessac-medium.onnx"))

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


@dataclass
class GuildSession:
    text_channel_id: int
    speak_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


sessions: dict[int, GuildSession] = {}

intents = discord.Intents.default()
intents.guilds = True
intents.message_content = True
intents.voice_states = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)


def configure_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "").endswith("bot.log") for h in root.handlers):
        return

    file_handler = RotatingFileHandler("bot.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
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
            _local_tts_voice = PiperVoice.load(LOCAL_TTS_MODEL_PATH)
        return _local_tts_voice


def resolve_tts_backend() -> str:
    if TTS_BACKEND in {"piper", "edge"}:
        return TTS_BACKEND
    if LOCAL_TTS_MODEL_PATH.exists() and PiperVoice is not None:
        return "piper"
    return "edge"


async def get_http_session() -> aiohttp.ClientSession:
    global _http_session
    if _http_session is None or _http_session.closed:
        timeout = aiohttp.ClientTimeout(total=45, connect=10)
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=25, ttl_dns_cache=300)
        _http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    return _http_session


async def health_handler(_request: web.Request) -> web.Response:
    return web.json_response(
        {
            "status": "ok",
            "bot_ready": bot.is_ready(),
            "bot_user": str(bot.user) if bot.user else None,
            "active_voice_sessions": len(sessions),
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
            vc = await channel.connect(timeout=20)
        except asyncio.TimeoutError:
            await ctx.send("Failed to connect (timeout).")
            return None
    else:
        vc = ctx.voice_client

    return vc


async def generate_ai_reply(prompt: str, guild: discord.Guild, speaker: discord.Member) -> str:
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
    candidate_tokens = [AI_MAX_TOKENS, 2048, 1024, 512, 256]
    # Preserve order while removing duplicates and invalid values.
    token_limits = []
    for value in candidate_tokens:
        if value > 0 and value not in token_limits:
            token_limits.append(value)

    text = ""
    last_error = None
    for token_limit in token_limits:
        text = ""
        body = {
            "model": AI_MODEL,
            "temperature": AI_TEMPERATURE,
            "max_tokens": token_limit,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a Discord voice assistant. Give clear, well-structured responses with good grammar. "
                        "Keep default replies concise for fast speech, and provide more detail only when explicitly asked."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Server: {guild.name}\nSpeaker: {speaker.display_name}\nPrompt: {prompt}",
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


async def synthesize_speech(text: str) -> Path:
    output_dir = Path("recordings") / "tts"
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = resolve_tts_backend()
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
    await edge_tts.Communicate(text, voice=TTS_VOICE, rate=TTS_RATE).save(str(output_path))
    return output_path


async def speak_text(vc: discord.VoiceClient, session_state: GuildSession, text: str) -> None:
    spoken = text.strip()
    if not spoken:
        return

    if VOICE_REPLY_MODE == "brief":
        # Faster voice response: speak only the first concise segment while keeping full text in chat.
        first = re.split(r"(?<=[.!?])\s+", spoken, maxsplit=1)[0].strip()
        spoken = first or spoken

    char_limit = min(MAX_TTS_CHARS, VOICE_MAX_CHARS) if VOICE_REPLY_MODE == "brief" else MAX_TTS_CHARS
    if len(spoken) > char_limit:
        spoken = spoken[:char_limit].rsplit(" ", 1)[0].strip() + "..."

    async with session_state.speak_lock:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is not installed on the host, so voice playback cannot start.")

        audio_path = await synthesize_speech(spoken)
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

    try:
        response = await generate_ai_reply(prompt, message.guild, message.author)
    except Exception as exc:
        log.exception("AI request failed for prompt: %r", prompt)
        response = "I hit a temporary AI connection issue. Please try again in a moment."

    # Always send a reply so the bot never appears unresponsive.
    await message.reply(response)

    try:
        await speak_text(vc, session_state, response)
    except Exception as exc:
        log.exception("Voice playback failed for prompt: %r", prompt)
        await message.channel.send(f"Text reply sent, but voice playback failed: {exc}")


@bot.event
async def on_ready() -> None:
    log.info("Bot ready as %s", bot.user)
    if resolve_tts_backend() == "piper":
        try:
            await asyncio.to_thread(get_local_tts_voice)
            log.info("Local piper voice preloaded")
        except Exception:
            log.exception("Failed to preload local piper voice")


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
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

    sessions[ctx.guild.id] = GuildSession(text_channel_id=ctx.channel.id)
    await ctx.send(
        f"Connected to {vc.channel.name}. Mention with {MENTION_TRIGGER} and I will reply in text + voice."
    )


@bot.command()
async def leave(ctx: commands.Context) -> None:
    vc = ctx.voice_client
    sessions.pop(ctx.guild.id, None)

    if not vc:
        await ctx.send("I am not in a voice channel.")
        return

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
        attempts = 0
        while True:
            attempts += 1
            try:
                await bot.start(token)
                break
            except discord.HTTPException as exc:
                status = getattr(exc, "status", None)
                text = str(exc).lower()
                is_rate_limited = status == 429 or "error 1015" in text or "rate limited" in text
                if not is_rate_limited:
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
