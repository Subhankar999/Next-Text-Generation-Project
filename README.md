# 🧠 LSTM Next Word Prediction & Sentence Generation

A web app that predicts the next word, or generates a whole sentence, from a starting phrase.
A **PyTorch LSTM** does the predicting, a **FastAPI** backend serves it, and a **Streamlit** frontend gives it a clean, themeable interface.

> **Live demo:** https://next-text-generation-project-ksa4fgzfekepfuaemuywt3.streamlit.app/


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



## 🧰 Tech stack

PyTorch · FastAPI · Uvicorn · Keras Tokenizer · Streamlit · Requests · Render · Streamlit Community Cloud

## ⚠️ Limitations

- The model always picks the single most likely next word, so generated text can be repetitive.
- The vocabulary is limited to the training data (a quotes dataset).
- History is stored per browser session and is not saved permanently.


