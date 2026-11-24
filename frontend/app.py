"""Streamlit frontend for the LSTM Next Word Prediction API."""

from __future__ import annotations

import html
import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent


# ------------------------------ CONFIG --------------------------------

CONFIG_PATH = BASE_DIR / "config.json"

@st.cache_data
def load_config(file_mtime: float) -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

CFG = load_config(CONFIG_PATH.stat().st_mtime)


CFG = load_config()

DEFAULT_THEME = {
    "primary": "#E8506E",
    "accent": "#E8710A",
    "bg_start": "#FFF8F0",
    "bg_end": "#FFEFE3",
    "card_background": "#FFFFFF",
    "card_border": "#F6DCCB",
    "text": "#2D2A32",
    "text_muted": "#8A7F87",
    "sidebar_bg": "#FFEFE3",
    "input_bg": "#FFFFFF",
}

# Theme presets live in config.json -> "themes". The user picks one in the sidebar.
THEMES = CFG.get("themes") or {"Default": CFG.get("theme", {})}
THEME_NAMES = list(THEMES)
st.session_state.setdefault("theme_name", CFG.get("default_theme", THEME_NAMES[0]))
if st.session_state.theme_name not in THEMES:
    st.session_state.theme_name = THEME_NAMES[0]
T = {**DEFAULT_THEME, **THEMES[st.session_state.theme_name]}
MSG = CFG["messages"]
APP = CFG["app"]
API_CFG = CFG["api"]
HISTORY = CFG.get("history", {"enabled": True, "max_items": 5})
EXAMPLES = CFG.get("examples", [])
FEATURES = {
    "clear_button": True,
    "add_word_button": True,
    "continue_button": True,
    "download_button": True,
    "show_response_time": True,
    "history_reuse": True,
    "theme_switcher": True,
    **CFG.get("features", {}),
}

st.set_page_config(
    page_title=APP["title"],
    page_icon=APP["icon"],
    layout=APP.get("layout", "centered"),
    initial_sidebar_state="expanded",
)


def esc(value) -> str:
    """Escape user/model text before putting it inside raw HTML."""
    return html.escape(str(value if value is not None else ""))


def get_api_url() -> str | None:
    """Read API URL from Streamlit secrets or environment variable."""
    try:
        url = st.secrets.get("API_URL")
    except Exception:
        url = None
    if not url:
        url = os.getenv("API_URL")
    return url.strip().rstrip("/") if url else None


API_URL = get_api_url()


# ------------------------------- STYLES --------------------------------

st.markdown(
    f"""
<style>
#MainMenu, footer {{visibility: hidden;}}
.stApp {{ background: linear-gradient(160deg, {T['bg_start']}, {T['bg_end']}); color: {T['text']}; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: {T['sidebar_bg']}; }}
.stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3,
[data-testid="stWidgetLabel"] p {{ color: {T['text']}; }}
textarea, [data-baseweb="textarea"], [data-baseweb="select"] > div {{
    background-color: {T['input_bg']} !important; color: {T['text']} !important;
}}
[data-testid="stCode"] pre, [data-testid="stCode"] code {{
    background: {T['input_bg']} !important; color: {T['text']} !important;
}}
.block-container {{ padding-top: 2rem; max-width: 820px; }}
.hero {{
    text-align: center; padding: 1.6rem 1rem 1.2rem; border-radius: 18px;
    margin-bottom: 1.2rem; border: 1px solid {T['card_border']};
    background: linear-gradient(135deg, {T['primary']}33, {T['accent']}22);
}}
.hero h1 {{
    margin: 0; font-size: 2.1rem;
    background: linear-gradient(90deg, {T['primary']}, {T['accent']});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.hero p {{ color: {T['text_muted']}; margin: .4rem 0 0; }}
.card {{
    background: {T['card_background']}; border: 1px solid {T['card_border']};
    border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: .8rem;
    box-shadow: 0 4px 16px rgba(232, 80, 110, 0.08); color: {T['text']};
}}
.conf-top {{ display: flex; align-items: baseline; justify-content: space-between; }}
.conf-value {{ font-size: 2rem; font-weight: 700; color: {T['primary']}; }}
.conf-track {{ background: {T['card_border']}; border-radius: 999px; height: 14px; overflow: hidden; margin-top: .4rem; }}
.conf-fill {{ height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, {T['primary']}, {T['accent']}); }}
.card .label {{
    color: {T['text_muted']}; font-size: .75rem; letter-spacing: .08em;
    text-transform: uppercase; margin-bottom: .3rem;
}}
.card .value {{ font-size: 1.05rem; line-height: 1.6; word-wrap: break-word; }}
.big-word {{ font-size: 2.2rem; font-weight: 700; color: {T['accent']}; }}
.gen {{ color: {T['accent']}; font-weight: 600; }}
.footer {{ text-align: center; color: {T['text_muted']}; font-size: .8rem; margin-top: 2rem; }}
div.stButton > button, div.stDownloadButton > button {{
    width: 100%; border-radius: 12px; font-weight: 600; padding: .6rem 1rem;
    background: {T['card_background']}; color: {T['text']}; border: 1px solid {T['card_border']};
}}
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="stBaseButton-primary"] {{
    background: linear-gradient(90deg, {T['primary']}, {T['accent']});
    color: #fff; border: none;
}}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="stBaseButton-primary"]:hover {{ filter: brightness(1.07); color: #fff; }}
</style>
""",
    unsafe_allow_html=True,
)


# -------------------------------- HEADER -------------------------------

st.markdown(
    f"""
    <div class="hero">
        <h1>{esc(APP['icon'])} {esc(APP['title'])}</h1>
        <p>{esc(APP['subtitle'])}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Stop early (before anything uses API_URL) if the secret is missing.
if not API_URL:
    st.error(
        "API_URL is not configured. Add it to .streamlit/secrets.toml locally "
        "or to the app's Secrets settings on Streamlit Cloud."
    )
    st.stop()


# ------------------------------ API REQUEST ---------------------------

class ApiError(Exception):
    pass


def make_url(endpoint: str) -> str:
    return f"{API_URL}/{endpoint.lstrip('/')}"


def call_api(endpoint: str, payload: dict) -> dict:
    """POST JSON to the FastAPI backend, with optional retries."""
    url = make_url(endpoint)
    attempts = API_CFG.get("retries", 1) + 1
    last_error: ApiError | None = None

    for attempt in range(attempts):
        try:
            response = requests.post(url, json=payload, timeout=API_CFG["timeout_seconds"])

            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                if isinstance(detail, list):  # FastAPI validation errors
                    detail = "; ".join(d.get("msg", "invalid input") for d in detail)
                if "no known words" in str(detail).lower():
                    raise ApiError(MSG.get("no_known_words", str(detail)))
                raise ApiError(f"Server returned {response.status_code}: {detail}")

            return response.json()

        except ApiError:
            raise  # server answered; retrying will not help
        except requests.exceptions.Timeout:
            last_error = ApiError(MSG["timeout_error"])
        except requests.exceptions.ConnectionError:
            last_error = ApiError(MSG["connection_error"])
        except ValueError:
            raise ApiError("The server sent an invalid response.")
        except requests.exceptions.RequestException as e:
            raise ApiError(f"Request failed: {e}")

        if attempt < attempts - 1:
            time.sleep(2)

    raise last_error or ApiError("API request failed.")


def parse_confidence(value) -> float:
    try:
        return float(str(value).strip().rstrip("%"))
    except ValueError:
        return 0.0


# ------------------------------ SESSION STATE -------------------------

st.session_state.setdefault("user_text", "")
st.session_state.setdefault("result", None)
st.session_state.setdefault("history", [])
st.session_state.setdefault("num_words", CFG["slider"]["default"])


def set_example(text: str):
    st.session_state.user_text = text[: CFG["input"]["max_chars"]]


def append_word(word: str):
    """Add the predicted word to the input so you can build a sentence word by word."""
    joined = f"{st.session_state.user_text.strip()} {word}".strip()
    st.session_state.user_text = joined[: CFG["input"]["max_chars"]]


def clear_all():
    st.session_state.user_text = ""
    st.session_state.result = None


def clear_history():
    st.session_state.history = []


def save_history(entry: dict):
    if HISTORY.get("enabled"):
        st.session_state.history.insert(0, entry)
        del st.session_state.history[HISTORY.get("max_items", 5):]


# -------------------------------- EXAMPLES -----------------------------

if EXAMPLES:
    st.markdown("**Try an example**")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        cols[i % 2].button(ex, key=f"ex_{i}", on_click=set_example, args=(ex,))


# -------------------------------- INPUT --------------------------------

st.text_area(
    CFG["input"]["label"],
    key="user_text",
    placeholder=CFG["input"]["placeholder"],
    height=CFG["input"].get("height", 120),
    max_chars=CFG["input"]["max_chars"],
)

s = CFG["slider"]
st.slider(s["label"], s["min"], s["max"], key="num_words", step=s["step"])

b1, b2 = st.columns(2)
predict_word = b1.button(CFG["buttons"]["word"], type="primary")
generate_sentence = b2.button(CFG["buttons"]["sentence"], type="primary")

if FEATURES["clear_button"]:
    st.button(" Clear input and result", on_click=clear_all)


# ------------------------------ PREDICTION -----------------------------

text = st.session_state.user_text.strip()

if predict_word or generate_sentence:
    if not text:
        st.session_state.result = None
        st.warning(MSG["empty_input"])
    else:
        try:
            if predict_word:
                t0 = time.perf_counter()
                with st.spinner(MSG["loading_word"]):
                    data = call_api(API_CFG["word_endpoint"], {"text": text})
                elapsed = time.perf_counter() - t0

                st.session_state.result = {
                    "type": "word",
                    "input": data.get("Input text", text),
                    "word": data.get("Next possible word"),
                    "confidence": parse_confidence(data.get("Confidence", "0")),
                    "elapsed": elapsed,
                }
                save_history({
                    "kind": "Word",
                    "input": text,
                    "output": data.get("Next possible word") or "",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
            else:
                num_words = int(st.session_state.num_words)
                t0 = time.perf_counter()
                with st.spinner(MSG["loading_sentence"]):
                    data = call_api(
                        API_CFG["sentence_endpoint"],
                        {"text": text, "num_words": num_words},
                    )
                elapsed = time.perf_counter() - t0

                st.session_state.result = {
                    "type": "sentence",
                    "input": text,
                    "output": data.get("Predicted text", ""),
                    "num_words": num_words,
                    "elapsed": elapsed,
                }
                save_history({
                    "kind": "Sentence",
                    "input": text,
                    "output": data.get("Predicted text", ""),
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

        except ApiError as e:
            st.session_state.result = None
            st.error(f"❌ {e}")


# -------------------------------- RESULTS ------------------------------

res = st.session_state.result

if res:
    st.divider()

    st.markdown(
        f"""
        <div class="card">
            <div class="label">Input text</div>
            <div class="value">{esc(res['input'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if res["type"] == "word":
        if res["word"]:
            st.markdown(
                f"""
                <div class="card">
                    <div class="label">Next possible word</div>
                    <div class="big-word">{esc(res['word'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            pct = min(max(res["confidence"], 0.0), 100.0)
            st.markdown(
                f"""
                <div class="card">
                    <div class="conf-top">
                        <div class="label">Confidence</div>
                        <div class="conf-value">{pct:.2f}%</div>
                    </div>
                    <div class="conf-track"><div class="conf-fill" style="width:{pct:.2f}%"></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if pct < 20:
                st.caption("Low confidence: the model sees several plausible next words.")
            if FEATURES["show_response_time"]:
                st.caption(f"⏱ Response time: {res['elapsed']:.2f}s")
            if FEATURES["add_word_button"]:
                st.button("➕ Add word to input", on_click=append_word, args=(res["word"],))
        else:
            st.warning("The model could not find a word for this input.")

    else:
        generated = res["output"]
        if generated.startswith(res["input"]):
            new_part = generated[len(res["input"]):]
            body = f"{esc(res['input'])}<span class='gen'>{esc(new_part)}</span>"
        else:  # never hide the model's output
            new_part = generated
            body = f"<span class='gen'>{esc(generated)}</span>"

        st.markdown(
            f"""
            <div class="card">
                <div class="label">Generated text · {res['num_words']} words requested</div>
                <div class="value">{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.code(generated, language=None)

        a1, a2 = st.columns(2)
        if FEATURES["download_button"]:
            a1.download_button(
                "💾 Download as .txt",
                data=generated,
                file_name="generated_text.txt",
                mime="text/plain",
            )
        if FEATURES["continue_button"]:
            a2.button("↪️ Continue from this text", on_click=set_example, args=(generated,))

        words_added = len(new_part.split())
        if words_added < res["num_words"]:
            st.info(f"Generation stopped early after {words_added} word(s).")
        if FEATURES["show_response_time"]:
            st.caption(f"⏱ Response time: {res['elapsed']:.2f}s")


# -------------------------------- SIDEBAR ------------------------------
# Rendered last so the history list already includes the newest request.

with st.sidebar:
    if FEATURES["theme_switcher"] and len(THEME_NAMES) > 1:
        st.selectbox(" Theme", THEME_NAMES, key="theme_name")

    if HISTORY.get("enabled") and st.session_state.history:
        st.subheader("🕘 Recent")
        for i, h in enumerate(st.session_state.history):
            preview = h["input"][:40] + ("…" if len(h["input"]) > 40 else "")
            if FEATURES["history_reuse"]:
                st.button(
                    f"↩ {h['kind']} · {preview}",
                    key=f"hist_{i}",
                    on_click=set_example,
                    args=(h["input"],),
                    help=f"{h['time']} → {h['output'][:120]}",
                )
            else:
                st.caption(f"**{h['kind']}** · {preview}")

        st.button("🗑 Clear history", on_click=clear_history)


# -------------------------------- FOOTER -------------------------------

st.markdown(
    f"<div class='footer'>{esc(APP.get('footer', ''))}</div>",
    unsafe_allow_html=True,
)
