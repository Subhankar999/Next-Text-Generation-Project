"""Streamlit frontend for the LSTM Next Word Prediction API."""
import json
import os
import time
from pathlib import Path

import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent


# ------------------------------ Config --------------------------------
@st.cache_data
def load_config() -> dict:
    with open(BASE_DIR / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


CFG = load_config()
T = CFG["theme"]
MSG = CFG["messages"]

st.set_page_config(
    page_title=CFG["app"]["title"],
    page_icon=CFG["app"]["icon"],
    layout="centered",
)


def get_api_url() -> str | None:
    """API URL comes from Streamlit secrets (env var as local fallback)."""
    try:
        url = st.secrets["API_URL"]
    except Exception:
        url = os.getenv("API_URL")
    return url.rstrip("/") if url else None


API_URL = get_api_url()


# ------------------------------- Styles -------------------------------
st.markdown(
    f"""
<style>
#MainMenu, footer {{visibility: hidden;}}
.block-container {{padding-top: 2rem; max-width: 820px;}}
.hero {{
    text-align: center; padding: 1.6rem 1rem 1.2rem;
    border-radius: 18px; margin-bottom: 1.2rem;
    background: linear-gradient(135deg, {T['primary']}33, {T['accent']}22);
    border: 1px solid {T['card_border']};
}}
.hero h1 {{
    margin: 0; font-size: 2.1rem;
    background: linear-gradient(90deg, {T['primary']}, {T['accent']});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.hero p {{color: {T['text_muted']}; margin: .4rem 0 0;}}
.card {{
    background: {T['card_background']}; border: 1px solid {T['card_border']};
    border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: .8rem;
}}
.card .label {{
    color: {T['text_muted']}; font-size: .75rem; letter-spacing: .08em;
    text-transform: uppercase; margin-bottom: .3rem;
}}
.card .value {{font-size: 1.05rem; line-height: 1.6; word-wrap: break-word;}}
.big-word {{
    font-size: 2.2rem; font-weight: 700; color: {T['accent']};
}}
.gen {{color: {T['accent']}; font-weight: 600;}}
.footer {{text-align: center; color: {T['text_muted']}; font-size: .8rem; margin-top: 2rem;}}
div.stButton > button {{
    width: 100%; border-radius: 12px; font-weight: 600; padding: .6rem 1rem;
}}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""<div class="hero"><h1>{CFG['app']['icon']} {CFG['app']['title']}</h1>
<p>{CFG['app']['subtitle']}</p></div>""",
    unsafe_allow_html=True,
)


# ------------------------------ API layer -----------------------------
class ApiError(Exception):
    pass


def call_api(endpoint: str, payload: dict) -> dict:
    """POST JSON to the backend and return parsed JSON, raising ApiError on failure."""
    url = f"{API_URL}{endpoint}"
    attempts = CFG["api"]["retries"] + 1
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            resp = requests.post(url, json=payload, timeout=CFG["api"]["timeout_seconds"])
            if resp.status_code >= 400:
                try:
                    detail = resp.json().get("detail", resp.text)
                except ValueError:
                    detail = resp.text
                if isinstance(detail, list):  # FastAPI validation errors
                    detail = "; ".join(d.get("msg", "invalid input") for d in detail)
                raise ApiError(f"Server returned {resp.status_code}: {detail}")
            return resp.json()
        except requests.exceptions.Timeout as e:
            last_error = ApiError(MSG["timeout_error"])
        except requests.exceptions.ConnectionError as e:
            last_error = ApiError(MSG["connection_error"])
        except ValueError:
            raise ApiError("The server sent an invalid response.")
        except requests.exceptions.RequestException as e:
            raise ApiError(f"Request failed: {e}")
        if attempt < attempts - 1:
            time.sleep(2)  # short pause, then retry (helps with Render cold starts)

    raise last_error  # type: ignore[misc]


@st.cache_data(ttl=30, show_spinner=False)
def check_health(url: str) -> bool:
    try:
        r = requests.get(f"{url}{CFG['api']['health_endpoint']}", timeout=5)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


# ----------------------------- Session state --------------------------
st.session_state.setdefault("user_text", "")
st.session_state.setdefault("result", None)
st.session_state.setdefault("history", [])
st.session_state.setdefault("num_words", CFG["slider"]["default"])


def set_example(text: str):
    st.session_state.user_text = text


def save_history(entry: dict):
    if CFG["history"]["enabled"]:
        st.session_state.history.insert(0, entry)
        del st.session_state.history[CFG["history"]["max_items"]:]


# --------------------------------- UI ---------------------------------
if not API_URL:
    st.error(MSG["missing_secret"])
    st.stop()

with st.sidebar:
    st.subheader("⚙️ Status")
    if check_health(API_URL):
        st.success("API online")
    else:
        st.warning("API not responding (it may be waking up)")
    st.caption(f"Endpoint: `{API_URL}`")
    if st.button("Refresh status"):
        check_health.clear()
        st.rerun()

    if CFG["history"]["enabled"] and st.session_state.history:
        st.subheader("🕘 Recent")
        for h in st.session_state.history:
            st.caption(f"**{h['kind']}** · {h['input'][:40]}{'…' if len(h['input']) > 40 else ''}")

st.markdown("**Try an example**")
cols = st.columns(2)
for i, ex in enumerate(CFG["examples"]):
    cols[i % 2].button(ex, key=f"ex_{i}", on_click=set_example, args=(ex,))

st.text_area(
    CFG["input"]["label"],
    key="user_text",
    placeholder=CFG["input"]["placeholder"],
    height=CFG["input"]["height"],
    max_chars=CFG["input"]["max_chars"],
)

s = CFG["slider"]
st.slider(s["label"], s["min"], s["max"], key="num_words", step=s["step"])

b1, b2 = st.columns(2)
predict_word = b1.button(CFG["buttons"]["word"], type="primary")
generate_sentence = b2.button(CFG["buttons"]["sentence"], type="primary")

text = st.session_state.user_text.strip()

if predict_word or generate_sentence:
    if not text:
        st.warning(MSG["empty_input"])
    else:
        try:
            if predict_word:
                with st.spinner(MSG["loading_word"]):
                    data = call_api(CFG["api"]["word_endpoint"], {"text": text})
                conf = float(str(data.get("Confidence", "0")).rstrip("%"))
                st.session_state.result = {
                    "type": "word",
                    "input": data.get("Input text", text),
                    "word": data.get("Next possible word"),
                    "confidence": conf,
                }
                save_history({"kind": "Word", "input": text})
            else:
                with st.spinner(MSG["loading_sentence"]):
                    data = call_api(
                        CFG["api"]["sentence_endpoint"],
                        {"text": text, "num_words": st.session_state.num_words},
                    )
                st.session_state.result = {
                    "type": "sentence",
                    "input": text,
                    "output": data.get("Predicted text", ""),
                    "num_words": st.session_state.num_words,
                }
                save_history({"kind": "Sentence", "input": text})
        except ApiError as e:
            st.session_state.result = None
            st.error(f"❌ {e}")

# ------------------------------- Results ------------------------------
res = st.session_state.result
if res:
    st.divider()
    st.markdown(
        f"""<div class="card"><div class="label">Input text</div>
<div class="value">{res['input']}</div></div>""",
        unsafe_allow_html=True,
    )
    if res["type"] == "word":
        if res["word"]:
            st.markdown(
                f"""<div class="card"><div class="label">Next possible word</div>
<div class="big-word">{res['word']}</div></div>""",
                unsafe_allow_html=True,
            )
            st.markdown("**Confidence**")
            st.progress(min(max(res["confidence"] / 100, 0.0), 1.0))
            st.caption(f"{res['confidence']:.2f}% confident")
        else:
            st.warning("The model could not find a word for this input.")
    else:
        generated = res["output"]
        new_part = generated[len(res["input"]):] if generated.startswith(res["input"]) else ""
        st.markdown(
            f"""<div class="card"><div class="label">Generated text · {res['num_words']} words requested</div>
<div class="value">{res['input']}<span class="gen">{new_part}</span></div></div>""",
            unsafe_allow_html=True,
        )
        st.code(generated, language=None)
        words_added = len(new_part.split())
        if words_added < res["num_words"]:
            st.info(f"Generation stopped early after {words_added} word(s).")

st.markdown(f"""<div class="footer">{CFG['app']['footer']}</div>""", unsafe_allow_html=True)
