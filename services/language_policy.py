from __future__ import annotations


def is_sanskrit_language(language: str) -> bool:
    return language.strip().lower().replace("_", "-").split("-", 1)[0] == "sa"


def map_language_code(configured: str) -> str:
    normalized = configured.strip().replace("_", "-")
    if normalized.lower() in {"sa", "sa-in"}:
        return "sa-IN"
    language_map = {
        "en-US": "en-US",
        "en": "en-US",
        "de-DE": "de-DE",
        "de": "de-DE",
    }
    return language_map.get(normalized, "en-US")


def stt_provider_chain(
    primary: str,
    language: str,
    enable_google_fallback: bool,
    enable_sphinx_fallback: bool,
) -> list[str]:
    if primary == "offline_whisper" or is_sanskrit_language(language):
        return ["offline_whisper"]

    chain = [primary]
    if enable_google_fallback and "google" not in chain:
        chain.append("google")
    if enable_sphinx_fallback and "sphinx" not in chain:
        chain.append("sphinx")
    return chain


def tts_provider_chain(configured: list[str], language: str) -> list[str]:
    if is_sanskrit_language(language):
        return ["sanskrit_local"]
    allowed = {"sanskrit_local", "elevenlabs", "google_cloud", "amazon_polly", "gtts", "offline"}
    chain = [provider for provider in configured if provider in allowed]
    if "offline" not in chain:
        chain.append("offline")
    return chain
