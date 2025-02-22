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

# Download model from Google Drive
DRIVE_ID = "13W0sa_WmCtI47J3er4OvHpv0my3jaErt"

if not os.path.exists('best_model.keras'):
    print("Downloading model from Google Drive...")
    download_url = f"https://drive.google.com/uc?export=download&id={DRIVE_ID}"
    response = requests.get(download_url)
    with open('best_model.keras', 'wb') as f:
        f.write(response.content)

# Load model
print("Loading model...")
model = tf.keras.models.load_model('best_model.keras')
print("Model loaded successfully!")

# Define your classes
CLASSES = ['cats', 'dogs', 'humans', 'no_object']

class ImageRequest(BaseModel):
    bucket_name: str
    file_path: str

def preprocess_image(image):
    # Resize image to match your model's expected input size
    image = image.resize((224, 224))
    
    # Convert to array and preprocess
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.post("/predict")
async def predict(request: ImageRequest):
    try:
        # Get image from Supabase
        print(f"Getting image from bucket: {request.bucket_name}, path: {request.file_path}")
        url = supabase.storage.from_(request.bucket_name).get_public_url(request.file_path)
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
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
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "API is running", "message": "Welcome to the Classification API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
