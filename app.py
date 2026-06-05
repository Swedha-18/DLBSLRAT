from __future__ import annotations

import base64
import json as _json
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import nltk
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from collection_engine import DataCollectionEngine
from sign_engine import SignLanguageEngine

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cnn8grps_rad1_model.h5"
ASSETS_DIR = BASE_DIR / "assets"

app = Flask(__name__)

SUPPORTED_TRANSLATE_LANGS = {
    "en": "English",
    "ta": "Tamil",
    "te": "Telugu",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "pa": "Punjabi",
}


# ---------------------------------------------------------------------------
# Translation helper
# Primary: unofficial Google Translate API (no key, good Tamil/Indian support)
# Fallback: MyMemory free API
# ---------------------------------------------------------------------------
def translate_text(text: str, source_lang: str, target_lang: str = "en") -> str:
    """Translate text. Returns original text if every provider fails."""
    if not text or source_lang == target_lang:
        return text
    if source_lang not in SUPPORTED_TRANSLATE_LANGS or target_lang not in SUPPORTED_TRANSLATE_LANGS:
        return text

    # ── Provider 1: unofficial Google Translate ──────────────────────────
    try:
        params = urllib.parse.urlencode({
            "client": "gtx",
            "sl": source_lang,
            "tl": target_lang,
            "dt": "t",
            "q": text,
        })
        url = f"https://translate.googleapis.com/translate_a/single?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read().decode())
            # data[0] is a list of [translated_chunk, original_chunk, ...]
            translated = "".join(
                chunk[0] for chunk in data[0] if chunk and chunk[0]
            )
            if translated:
                return translated
    except Exception:
        pass

    # ── Provider 2: MyMemory fallback ────────────────────────────────────
    _BAD_PREFIXES = ("PLEASE SELECT", "QUERY LENGTH", "MYMEMORY WARNING", "YOU USED")
    try:
        params = urllib.parse.urlencode({"q": text, "langpair": f"{source_lang}|{target_lang}"})
        url = f"https://api.mymemory.translated.net/get?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = _json.loads(resp.read().decode())
            translated = data.get("responseData", {}).get("translatedText", "")
            status = data.get("responseStatus", 0)
            if (
                translated
                and status == 200
                and not any(translated.upper().startswith(p) for p in _BAD_PREFIXES)
            ):
                return translated
    except Exception:
        pass

    return text

# ---------------------------------------------------------------------------
# Sign-to-Text engine
# ---------------------------------------------------------------------------
engine = SignLanguageEngine(model_path=str(MODEL_PATH))
collector_engine = DataCollectionEngine()


def _decode_image(image_data):
    if not image_data:
        return None
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    try:
        raw = base64.b64decode(image_data)
    except Exception:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Text-to-Sign NLP helpers
# ---------------------------------------------------------------------------
STOP_WORDS = {
    "a", "ain", "am", "an", "are", "aren", "aren't", "as", "be", "been",
    "being", "couldn", "couldn't", "d", "did", "didn", "didn't", "do",
    "does", "doesn", "doesn't", "don", "don't", "for", "hadn", "hadn't",
    "has", "hasn", "hasn't", "haven", "haven't", "having", "i", "is",
    "isn", "isn't", "itself", "ll", "m", "ma", "mightn", "mightn't",
    "mustn", "mustn't", "needn", "needn't", "nor", "o", "off", "re", "s",
    "shan", "shan't", "she's", "shouldn", "shouldn't", "should've", "such",
    "t", "that", "that'll", "the", "then", "ve", "was", "wasn", "wasn't",
    "were", "weren", "weren't", "whom", "wouldn", "wouldn't", "y",
    "you'd", "you'll", "you're", "you've", "won't",
}

NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]

_nltk_ready = False


def ensure_nltk_resources() -> None:
    global _nltk_ready
    if _nltk_ready:
        return
    for resource_path, package_name in NLTK_RESOURCES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(package_name, quiet=True)
            except Exception:
                pass
    _nltk_ready = True


def resolve_token(token: str) -> str | None:
    token = token.strip()
    if not token:
        return None
    candidates = [token, token.title(), token.upper(), token.lower()]
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (ASSETS_DIR / f"{candidate}.mp4").exists():
            return candidate
    return None


def lemmatize_word(word: str, pos_tag: str, lemmatizer: WordNetLemmatizer) -> str:
    if pos_tag in {"VBG", "VBD", "VBZ", "VBN", "NN"}:
        return lemmatizer.lemmatize(word, pos="v")
    if pos_tag in {"JJ", "JJR", "JJS", "RBR", "RBS"}:
        return lemmatizer.lemmatize(word, pos="a")
    return lemmatizer.lemmatize(word)


def to_sign_sequence(text: str) -> list[str]:
    ensure_nltk_resources()
    try:
        words = word_tokenize(text)
    except LookupError:
        words = text.split()
    if not words:
        return []
    try:
        tagged = nltk.pos_tag(words)
    except LookupError:
        tagged = [(word, "NN") for word in words]

    tense = {
        "future": len([w for w in tagged if w[1] == "MD"]),
        "present": len([w for w in tagged if w[1] in {"VBP", "VBZ", "VBG"}]),
        "past": len([w for w in tagged if w[1] in {"VBD", "VBN"}]),
        "present_continuous": len([w for w in tagged if w[1] == "VBG"]),
    }

    lemmatizer = WordNetLemmatizer()
    filtered_words: list[str] = []
    for word, pos in tagged:
        if not any(ch.isalnum() for ch in word):
            continue
        if word.lower() in STOP_WORDS:
            continue
        base_word = lemmatize_word(word, pos, lemmatizer)
        filtered_words.append("Me" if base_word.lower() == "i" else base_word)

    if not filtered_words:
        return []

    probable_tense = max(tense, key=tense.get)
    words_with_tense = filtered_words.copy()

    if probable_tense == "past" and tense["past"] >= 1:
        words_with_tense.insert(0, "Before")
    elif probable_tense == "future" and tense["future"] >= 1:
        if all(w.lower() != "will" for w in words_with_tense):
            words_with_tense.insert(0, "Will")
    elif probable_tense == "present" and tense["present_continuous"] >= 1:
        words_with_tense.insert(0, "Now")

    sign_tokens: list[str] = []
    for word in words_with_tense:
        resolved_word = resolve_token(word)
        if resolved_word:
            sign_tokens.append(resolved_word)
            continue
        for char in word:
            if not char.isalnum():
                continue
            resolved_char = resolve_token(char)
            if resolved_char:
                sign_tokens.append(resolved_char)

    return sign_tokens


# ---------------------------------------------------------------------------
# Routes - Pages
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("sign_to_text.html")


@app.route("/sign-to-text")
def sign_to_text():
    return render_template("sign_to_text.html")


@app.route("/text-to-sign", methods=["GET", "POST"])
def text_to_sign():
    text = ""
    original_text = ""
    source_lang = "en"
    words: list[str] = []
    if request.method == "POST":
        text = (request.form.get("sen") or "").strip()
        original_text = (request.form.get("original_text") or text).strip()
        source_lang = (request.form.get("lang") or "en").strip().lower()
        if source_lang not in SUPPORTED_TRANSLATE_LANGS:
            source_lang = "en"
        if text:
            words = to_sign_sequence(text)
    return render_template(
        "text_to_sign.html",
        text=text,
        original_text=original_text,
        source_lang=source_lang,
        words=words,
    )


@app.route("/settings")
def settings():
    return render_template("settings.html")


@app.route("/predictor")
def predictor():
    return render_template("predictor.html")


@app.route("/collector")
def collector():
    return render_template("collector.html")


# ---------------------------------------------------------------------------
# Routes - Assets (sign language videos)
# ---------------------------------------------------------------------------
@app.route("/assets/<path:filename>")
def serve_asset(filename):
    return send_from_directory(str(ASSETS_DIR), filename)


# ---------------------------------------------------------------------------
# Routes - Sign-to-Text API
# ---------------------------------------------------------------------------
@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    frame = _decode_image(payload.get("image", ""))
    if frame is None:
        return jsonify({"error": "Invalid image payload."}), 400
    state = engine.process_frame(frame)
    return jsonify(state)


@app.post("/api/clear")
def clear_sentence():
    return jsonify(engine.clear())


@app.post("/api/suggestion")
def apply_suggestion():
    payload = request.get_json(silent=True) or {}
    try:
        idx = int(payload.get("index", 1)) - 1
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid suggestion index."}), 400
    return jsonify(engine.apply_suggestion(idx))


@app.post("/api/translate")
def api_translate():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    source = (payload.get("source") or "en").strip().lower()[:10]
    target = (payload.get("target") or "en").strip().lower()[:10]
    if not text:
        return jsonify({"error": "No text provided."}), 400
    if len(text) > 500:
        return jsonify({"error": "Text too long (max 500 chars)."}), 400
    if source not in SUPPORTED_TRANSLATE_LANGS:
        return jsonify({"error": f"Unsupported source language: {source}"}), 400
    if target not in SUPPORTED_TRANSLATE_LANGS:
        return jsonify({"error": f"Unsupported target language: {target}"}), 400
    translated = translate_text(text, source, target)
    return jsonify({"translated": translated, "source": source, "target": target})


@app.post("/api/collector/process")
def collector_process():
    payload = request.get_json(silent=True) or {}
    frame = _decode_image(payload.get("image", ""))
    if frame is None:
        return jsonify({"error": "Invalid image payload."}), 400
    modes = payload.get("modes", ["skeleton"])
    if isinstance(modes, str):
        modes = [modes]
    state = collector_engine.process_frame(
        frame_bgr=frame,
        label=payload.get("label", "A"),
        dataset_dir=payload.get("dataset_dir", "collected_data"),
        save=bool(payload.get("save", False)),
        modes=modes,
    )
    return jsonify(state)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
