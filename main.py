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

# Get the absolute path of the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.keras')

print(f"Current working directory: {os.getcwd()}")
print(f"Base directory: {BASE_DIR}")
print(f"Model path: {MODEL_PATH}")

def download_file_from_google_drive(file_id, destination):
    try:
        # First get a download URL that handles the virus scan warning
        URL = "https://drive.google.com/uc"
        session = requests.Session()
        
        # Get the initial response
        response = session.get(URL, params={
            'id': file_id,
            'export': 'download'
        }, stream=True)
        
        # If there's a download warning, we need to handle it
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'id': file_id, 'confirm': value, 'export': 'download'}
                response = session.get(URL, params=params, stream=True)
                break
        
        # Write the file in chunks
        print(f"Starting download to {destination}")
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
        
        # Verify the file was created
        if os.path.exists(destination):
            size = os.path.getsize(destination)
            print(f"Download completed. File size: {size} bytes")
            return True
        return False
        
    except Exception as e:
        print(f"Download error: {str(e)}")
        return False

app = FastAPI()

# Enable CORS
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

# Model initialization
DRIVE_ID = "13W0sa_WmCtI47J3er4OvHpv0my3jaErt"
CLASSES = ['cats', 'dogs', 'humans', 'no_object']

# Download and load model
print("Checking for model file...")
if not os.path.exists(MODEL_PATH):
    print("Model not found. Downloading from Google Drive...")
    success = download_file_from_google_drive(DRIVE_ID, MODEL_PATH)
    if not success:
        raise RuntimeError("Failed to download model from Google Drive")

# Verify model file exists and has content
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

file_size = os.path.getsize(MODEL_PATH)
if file_size == 0:
    raise ValueError(f"Model file is empty (0 bytes)")

print(f"Loading model from {MODEL_PATH} (size: {file_size} bytes)")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

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
    model_exists = os.path.exists(MODEL_PATH)
    model_size = os.path.getsize(MODEL_PATH) if model_exists else 0
    
    return {
        "status": "API is running",
        "message": "Welcome to the Classification API",
        "model_path": MODEL_PATH,
        "model_exists": model_exists,
        "model_size": model_size,
        "directory_contents": os.listdir(BASE_DIR)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
