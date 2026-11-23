from fastapi import FastAPI , HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import io 
import torch
import torch.nn as nn
import joblib 
import pandas as pd 
import numpy as np  
import pickle
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
app=FastAPI()

app.add_middleware(CORSMiddleware,
allow_origins=["*"],
allow_credentials=False,
allow_methods=["*"],
allow_headers=["*"]
)

# RNN model 
class lstm(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.lstm = nn.LSTM(embedding_dim, rnn_units, batch_first=True).to(device)
        self.fc = nn.Linear(rnn_units, vocab_size).to(device)

    def forward(self, input):
        out = self.embedding(input.long())
        out, hidden = self.lstm(out)

        # hidden[-1] has shape [batch_size, rnn_units]
        return self.fc(out[:, -1, :])
# Loading the tokkenizer
with open('tokenizer.pkl','rb') as file:
    tokenizer=pickle.load(file)
    
    
vocab_size=10000
embedding_dim=50
rnn_units=128

model=lstm(vocab_size,embedding_dim,rnn_units).to(device)
# Loading the model 
stat_dict=torch.load('lstm_model.pth', map_location=device,weights_only=True)
model.load_state_dict(stat_dict)
model.eval()


# Word generation BAse model
class text_gen(BaseModel):
    text: str = Field(min_length=1)
    
# Sentense generation base model
class sentence_gen(BaseModel):
    text: str = Field(min_length=1)
    num_words: int = Field(default=10, ge=1, le=100)   
    


@app.get("/")
def home():
    return {
        "Project" :"Next text generation",
        "Message": "This  is a LSTM project that can predict the possible next word for a sentense "
    }

def predict_next_word(text: str):
    try:
        
        
        text=text.lower()
        text=text.translate(str.maketrans('','', string.punctuation))
        
        sequence = tokenizer.texts_to_sequences([text])
        
        if not sequence or not sequence[0]:
            raise ValueError("Input contains no known words")
        
        max_length = max(len(seq) for seq in sequence)
               
        x_padded=pad_sequences(sequence, maxlen=max_length, padding='pre')
        x_tensor = torch.tensor(x_padded,dtype=torch.long,device=device)
        with torch.no_grad():
            prediction=model(x_tensor)
            
            prob=torch.softmax(prediction, dim=1)
            
            prob_index= torch.argmax(prob,dim=1).item()
            confidence = prob[0, prob_index].item()
            
            predicted_word=tokenizer.index_word.get(prob_index)
            
            return predicted_word, confidence
    except Exception as e:
        raise ValueError(f"Next word prediction Failed {str(e)}")
@app.post("/word_predict")
def word_predict(tg: text_gen):

    try:
        word, confidence = predict_next_word(tg.text)

        return {
            "Input text": tg.text,
            "Next possible word": word,
            "Confidence": f"{confidence * 100:.2f}%"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction Failed: {str(e)}"
        )
       
@app.post("/sentense_predict")
def sentense_predict(sg:sentence_gen):
    try:
        text=sg.text
        n=sg.num_words
    
        for _ in range(n):
            word, confidence = predict_next_word(text)
            if word=="":
                break
            text+=" "+ word   
        return {
            "Predicted text":text
        }
    
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Failed : {str(e)}")
    
    
    