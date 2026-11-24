# 🧠 LSTM Next Word Prediction & Sentence Generation

A web app that predicts the next word, or generates a whole sentence, from a starting phrase.
A **PyTorch LSTM** does the predicting, a **FastAPI** backend serves it, and a **Streamlit** frontend gives it a clean, themeable interface.

> **Live demo:** _add your Streamlit app link here_
> **API docs:** _add `https://your-service.onrender.com/docs` here_

---

## ✨ Features

- 🔮 **Next-word prediction** with a confidence score and bar
- ✍️ **Sentence generation** with 1 to 100 words, chosen with a slider
- ➕ **Add word to input** to build a sentence one word at a time
- ↪️ **Continue from this text** to keep extending generated output
- 💾 **Download** generated text as a `.txt` file
- 🕘 **History** in the sidebar, with one-click reuse and a clear button
- 🎨 **Theme switcher** with 4 presets (Sunset, Mint, Lavender, Midnight Dark)
- ⏱ Response time display
- 🛡️ Handling for empty input, loading states, timeouts, connection and server errors
- ⚙️ Whole UI (text, colors, limits, endpoints, feature toggles) controlled from `config.json`

## 🏗️ Architecture

```
┌────────────────────┐   POST (JSON)    ┌────────────────────────┐
│  Streamlit (UI)    │ ───────────────► │  FastAPI + PyTorch LSTM│
│  Streamlit Cloud   │ ◄─────────────── │  Render                │
└────────────────────┘      JSON        └────────────────────────┘
       reads API_URL from Streamlit Secrets
```

## 📁 Project structure

```
lstm-next-word-app/
├── README.md
├── render.yaml                     # optional Render blueprint
├── .gitignore
├── backend/
│   ├── app.py                      # FastAPI app + LSTM inference
│   ├── lstm_model.pth              # trained weights
│   ├── tokenizer.pkl               # trained tokenizer
│   └── requirements.txt
└── frontend/
    ├── streamlit_app.py            # Streamlit UI
    ├── config.json                 # UI settings, themes, feature toggles
    ├── requirements.txt
    └── .streamlit/
        ├── config.toml             # base Streamlit theme
        └── secrets.toml.example    # template for API_URL
```

## 🔌 API reference

| Method | Endpoint | Body | Response |
|---|---|---|---|
| GET | `/` | none | project info |
| GET | `/health` | none | `{"status": "ok"}` |
| POST | `/word_predict` | `{"text": "happiness is found in"}` | `{"Input text": "...", "Next possible word": "...", "Confidence": "12.34%"}` |
| POST | `/sentense_predict` | `{"text": "happiness is found in", "num_words": 10}` | `{"Predicted text": "..."}` |

`num_words` accepts 1 to 100. The endpoint name `/sentense_predict` is spelled this way on purpose, so existing behavior is preserved.
Interactive docs are available at `/docs` when the backend is running.

## 🚀 Run locally

**Requirements:** Python 3.10 or 3.11.

**1. Clone**
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

**2. Start the backend** (terminal 1)
```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```
Check it at http://127.0.0.1:8000/docs.

**3. Start the frontend** (terminal 2)
```bash
cd frontend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml      # Windows: copy ...
streamlit run streamlit_app.py
```
`secrets.toml` should contain:
```toml
API_URL = "http://127.0.0.1:8000"
```
Open http://localhost:8501. Both servers must be running at the same time.

## ☁️ Deployment

### Backend on Render
1. Push the repo to GitHub.
2. In Render: **New → Web Service**, connect the repo.
3. Settings:
   - **Root Directory:** `backend`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Environment variable:** `PYTHON_VERSION = 3.11.9`
   - **Health Check Path:** `/health`
4. Deploy, then open `https://<your-service>.onrender.com/health`. You should see `{"status":"ok"}`.

(Alternatively choose **New → Blueprint** and let Render read `render.yaml`.)

### Frontend on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
2. Pick the repo and branch, and set **Main file path** to `frontend/streamlit_app.py`.
3. Open **Advanced settings → Secrets** and add:
   ```toml
   API_URL = "https://<your-service>.onrender.com"
   ```
   Use the base URL only, with no `/docs` or `/health` at the end.
4. Deploy.

## 🎛️ Customizing with `config.json`

| Section | What it controls |
|---|---|
| `app` | title, icon, subtitle, footer |
| `themes` / `default_theme` | color presets shown in the sidebar theme picker |
| `examples` | one-click example sentences |
| `input` | label, placeholder, height, max characters |
| `slider` | word-count range and default |
| `buttons` | button labels |
| `features` | turn features on or off (`clear_button`, `add_word_button`, `continue_button`, `download_button`, `show_response_time`, `history_reuse`, `theme_switcher`) |
| `history` | enable it and set how many items are kept |
| `api` | endpoint paths, timeout, retries |
| `messages` | all user-facing messages |

To add a theme, copy any block in `themes`, rename it and change the colors. Edits to `config.json` are picked up automatically; after pushing to GitHub, wait about a minute and hard-refresh the browser.

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `API_URL is not configured` | Create `frontend/.streamlit/secrets.toml` locally, or add the secret in Streamlit Cloud. Restart the app afterwards. |
| `Cannot reach the backend` | The backend isn't running, or `API_URL` is wrong. |
| First request is slow (about 1 minute) | Render's free tier sleeps when idle. This is normal. |
| Backend crashes on Render (out of memory) | PyTorch plus TensorFlow can exceed the free 512 MB. Use a larger plan. |
| `tokenizer.pkl` fails to load | Pin the Keras/TensorFlow versions used when the tokenizer was saved. |
| `None of those words are in the model's vocabulary` | Try more common words. |
| `config.json` edits don't show | Check that you edited `frontend/config.json`, wait for the redeploy, then hard-refresh with `Ctrl + Shift + R`. |

## 🧰 Tech stack

PyTorch · FastAPI · Uvicorn · Keras Tokenizer · Streamlit · Requests · Render · Streamlit Community Cloud

## ⚠️ Limitations

- The model always picks the single most likely next word, so generated text can be repetitive.
- The vocabulary is limited to the training data (a quotes dataset).
- History is stored per browser session and is not saved permanently.

## 📄 License

Add your license here (for example MIT).
