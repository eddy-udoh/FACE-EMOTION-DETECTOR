"""
Facial Emotion Recognition Web Application
Flask app that detects emotions from uploaded images
(TFLite Version - Optimized for Render)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import cv2

# --- TFLITE CHANGE 1: IMPORT TFLITE RUNTIME ---
# We import the lightweight interpreter instead of the heavy Keras
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite
# ----------------------------------------------


# ============================================================================
# CONFIGURATION
# ============================================================================

app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_in_production'

def init_db():
    """Initialize the database and create tables if they don't exist"""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                age INTEGER NOT NULL,
                emotion TEXT NOT NULL,
                image_path TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        print("✓ Database initialized")
    except Exception as e:
        print(f"Database init error: {e}")

# Call the init function to make sure the DB exists
init_db()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

EMOTION_RESPONSES = {
    'angry': "You look angry. What's bothering you? Take a deep breath! 😤",
    'disgust': "You seem disgusted. Did something unpleasant happen? 🤢",
    'fear': "You look fearful. Everything will be okay! Stay strong! 😨",
    'happy': "You're happy! Keep smiling, it looks great on you! 😊",
    'sad': "You seem sad. Remember, tough times don't last but tough people do! 😢",
    'surprise': "You look surprised! What caught you off guard? 😲",
    'neutral': "You have a neutral expression. Feeling calm today? 😐"
}

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def save_to_database(name, email, age, emotion, image_path):
    """Save user data to the database"""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO users (name, email, age, emotion, image_path, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, email, age, emotion, image_path, datetime.now()))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Database error: {e}")
        return False

# ============================================================================
# MODEL LOADING (TFLite Version)
# ============================================================================

# --- TFLITE CHANGE 2: LOAD TFLITE MODEL ---
# This is much lighter and faster than loading the .h5 file
def load_emotion_model():
    """Load the trained TFLite model and allocate tensors"""
    try:
        model_path = os.path.join(os.getcwd(), 'face_emotionModel.tflite')
        if not os.path.exists(model_path):
            print("❌ Model file not found:", model_path)
            return None
        
        # Load TFLite model
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()  # Allocate memory
        
        print("✓ TFLite Model loaded successfully")
        return interpreter
    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        return None

emotion_model = load_emotion_model()
# ----------------------------------------------

# ============================================================================
# IMAGE PROCESSING & PREDICTION (TFLite Version)
# ============================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for emotion detection"""
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized / 255.0
        
        # --- TFLITE CHANGE 3: MATCH INPUT TYPE ---
        # TFLite models are often int8 or float32. We'll use float32.
        # We also need to add the batch dimension.
        final_image = np.expand_dims(normalized, axis=0) # (1, 48, 48)
        final_image = np.expand_dims(final_image, axis=-1) # (1, 48, 48, 1)
        
        # Ensure it's the correct type (float32)
        return final_image.astype(np.float32)
        # -------------------------------------------
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_emotion(image_path):
    """Predict emotion from an image using TFLite"""
    if emotion_model is None:
        print("❌ Model is not loaded.")
        return None, 0
    try:
        processed_image = preprocess_image(image_path)
        if processed_image is None:
            return None, 0

        # --- TFLITE CHANGE 4: RUNNING PREDICTION ---
        # Get input and output tensor details
        input_details = emotion_model.get_input_details()
        output_details = emotion_model.get_output_details()
        
        # Set the input tensor
        emotion_model.set_tensor(input_details[0]['index'], processed_image)
        
        # Run inference
        emotion_model.invoke()
        
        # Get the output tensor
        predictions = emotion_model.get_tensor(output_details[0]['index'])
        # -------------------------------------------
        
        # Process the results (same as before)
        emotion_index = np.argmax(predictions[0])
        confidence = predictions[0][emotion_index]
        emotion_label = EMOTIONS[emotion_index]
        
        return emotion_label, float(confidence)
        
    except Exception as e:
        print(f"Error predicting emotion: {e}")
        return None, 0

# ============================================================================
# ROUTES (No changes needed here)
# ============================================================================

@app.route('/')
def index():
    # Make sure your HTML file is 'index.htm' as per your project structure
    return render_template('index.htm') 

@app.route('/submit', methods=['POST'])
def submit():
    name = request.form.get('name', '').strip()
    email = request.form.get('email', '').strip()
    age = request.form.get('age', '').strip()

    if not name or not email or not age:
        flash('Please fill in all fields!', 'error')
        return redirect(url_for('index'))

    try:
        age = int(age)
        if age < 1 or age > 150:
            flash('Please enter a valid age!', 'error')
            return redirect(url_for('index'))
    except ValueError:
        flash('Age must be a number!', 'error')
        return redirect(url_for('index'))

    if 'image' not in request.files:
        flash('No image uploaded!', 'error')
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        flash('No image selected!', 'error')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Invalid file type! Please upload PNG, JPG, JPEG, or GIF.', 'error')
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    emotion, confidence = predict_emotion(filepath)
    if emotion is None:
        flash('Error processing image. Please try again.', 'error')
        return redirect(url_for('index'))

    success = save_to_database(name, email, age, emotion, filepath)
    if not success:
        flash('Error saving data. Please try again.', 'error')
        return redirect(url_for('index'))

    response_message = EMOTION_RESPONSES.get(emotion, "Emotion detected!")
    
    result_message = f"""
    <h2>Detection Complete! 🎉</h2>
    <p><strong>Name:</strong> {name}</p>
    <p><strong>Email:</strong> {email}</p>
    <p><strong>Age:</strong> {age}</p>
    <p><strong>Detected Emotion:</strong> {emotion.upper()}</p>
    <p><strong>Confidence:</strong> {confidence*100:.2f}%</p>
    <p><strong>Message:</strong> {response_message}</p>
    <p>Your information has been saved to the database!</p>
    <br>
    <a href="/" style="padding: 10px 20px; background: #4CAF50; color: white; text-decoration: none; border-radius: 5px;">Submit Another</a>
    """

    return result_message

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)