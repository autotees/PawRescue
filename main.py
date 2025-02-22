# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from supabase import create_client
import gdown
import os

app = FastAPI()

# Download model from Google Drive on startup
MODEL_ID = "https://drive.google.com/file/d/13W0sa_WmCtI47J3er4OvHpv0my3jaErt/view?usp=sharing"  # Get this from your Drive share link
if not os.path.exists('best_model.keras'):
    gdown.download(f'https://drive.google.com/uc?id={MODEL_ID}', 'best_model.keras', quiet=False)

# Load model
model = tf.keras.models.load_model('best_model.keras')

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Supabase setup
supabase = create_client(
    os.environ.get("SUPABASE_URL"),
    os.environ.get("SUPABASE_KEY")
)

class ImageRequest(BaseModel):
    bucket_name: str
    file_path: str

def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust size to match your model
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(request: ImageRequest):
    try:
        # Get image from Supabase
        url = supabase.storage.from_(request.bucket_name).get_public_url(request.file_path)
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        
        # Preprocess
        processed_image = preprocess_image(image)
        
        # Predict
        prediction = model.predict(processed_image)
        class_index = np.argmax(prediction[0])
        confidence = float(prediction[0][class_index])
        
        # Your class names
        classes = ['cats', 'dogs','humans','no_object']  # Replace with your classes
        predicted_class = classes[class_index]
        
        return {
            "prediction": predicted_class,
            "confidence": float(confidence)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "API is running"}
