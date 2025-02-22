# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from supabase import create_client
import os
from fastapi.middleware.cors import CORSMiddleware
import gdown

# Get the absolute path of the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.keras')

print(f"Current working directory: {os.getcwd()}")
print(f"Base directory: {BASE_DIR}")
print(f"Model path: {MODEL_PATH}")

# Global variable for model
model = None

def ensure_model():
    """Ensure model is downloaded and loaded"""
    global model
    
    try:
        # Direct Google Drive download URL
        url = 'https://drive.google.com/uc?id=13W0sa_WmCtI47J3er4OvHpv0my3jaErt'
        
        print("Attempting to download model...")
        # Use gdown for reliable Google Drive downloads
        gdown.download(url, MODEL_PATH, quiet=False)
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model download failed - file not found at {MODEL_PATH}")
            
        file_size = os.path.getsize(MODEL_PATH)
        print(f"Model file size: {file_size} bytes")
        
        if file_size == 0:
            raise ValueError("Downloaded file is empty")
            
        print("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error in ensure_model: {str(e)}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            print(f"Removed failed model file from {MODEL_PATH}")
        return False

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model
    if model is None:
        success = ensure_model()
        if not success:
            raise RuntimeError("Failed to initialize model")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

CLASSES = ['cats', 'dogs', 'humans', 'no_object']

class ImageRequest(BaseModel):
    bucket_name: str
    file_path: str

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(request: ImageRequest):
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
        
    try:
        print(f"Getting image from bucket: {request.bucket_name}, path: {request.file_path}")
        url = supabase.storage.from_(request.bucket_name).get_public_url(request.file_path)
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        
        processed_image = preprocess_image(image)
        
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class_index])
        predicted_class = CLASSES[predicted_class_index]
        
        print(f"Prediction: {predicted_class}, Confidence: {confidence}")
        
        return {
            "success": True,
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2)
        }
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    global model
    return {
        "status": "API is running",
        "message": "Welcome to the Classification API",
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "model_size": os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0,
        "model_loaded": model is not None,
        "directory_contents": os.listdir(BASE_DIR)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
