from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import os

app = FastAPI()

# هذا هو المكان الذي سنقوم برفع النموذج الحقيقي فيه لاحقاً (سأشرحه في الخطوة التالية)
MODEL_PATH = "model.h5"

# بناء النموذج (كما دربته في Colab)
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(10, 1), return_sequences=True),
        tf.keras.layers.LSTM(50, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_model()

@app.get("/predict")
async def get_prediction():
    # محاكاة بيانات آخر 10 ساعات (يجب أن تأتي من موقعك مستقبلاً)
    last_10_hours = np.array([40, 42, 45, 50, 55, 60, 66, 72, 78, 85]).reshape((1, 10, 1))
    
    # القيام بالتوقع
    pred = model.predict(last_10_hours, verbose=0)[0][0]
    realistic_prediction = int(min(100, max(0, round(pred + 5))))
    
    return {"prediction": realistic_prediction}
