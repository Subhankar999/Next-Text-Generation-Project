LSTM Next Word Prediction & Sentence Generation

A web app that predicts the next word, or generates a whole sentence, from a starting phrase. A PyTorch LSTM does the predicting, a FastAPI backend serves it, and a Streamlit frontend gives it a clean, themeable interface.

Live demo: add your Streamlit app link here
API docs: add https://your-service.onrender.com/docs here

✨ Features
🔮 Next-word prediction with a confidence score and bar
✍️ Sentence generation with 1 to 100 words, chosen with a slider
➕ Add word to input to build a sentence one word at a time
↪️ Continue from this text to keep extending generated output
💾 Download generated text as a .txt file
🕘 History in the sidebar, with one-click reuse and a clear button
🎨 Theme switcher with 4 presets (Sunset, Mint, Lavender, Midnight Dark)
⏱ Response time display
🛡️ Handling for empty input, loading states, timeouts, connection and server errors
⚙️ Whole UI (text, colors, limits, endpoints, feature toggles) controlled from config.json
