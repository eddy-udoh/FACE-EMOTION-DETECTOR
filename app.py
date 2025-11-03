"""
Facial Emotion Recognition Web Application
Flask app that detects emotions from uploaded images
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU for Render

import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow import keras
from PIL import Image
import cv2

# ============================================================================
# CONFIGURATION
# ============================================================================

app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_in_production'  # For flash messages

# --- DATABASE INITIALIZATION MOVED HERE ---
# This runs *once* when the app starts, ensuring the DB is ready.
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
# ------------------------------------------

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create uploads folder if it doesn't exist
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
# (init_db is now above)

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
# MODEL LOADING
# ============================================================================

def load_emotion_model():
    """Load the trained emotion detection model"""
    try:
        model_path = os.path.join(os.getcwd(), 'face_emotionModel.h5')
        if not os.path.exists(model_path):
            print("❌ Model file not found:", model_path)
            return None
        model = keras.models.load_model(model_path, compile=False)
        print("✓ Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

emotion_model = load_emotion_model()

# ============================================================================
# IMAGE PROCESSING & PREDICTION
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
        final = normalized.reshape(1, 48, 48, 1)
        return final
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_emotion(image_path):
    """Predict emotion from an image"""
    if emotion_model is None:
        return None, 0
    try:
        processed_image = preprocess_image(image_path)
        if processed_image is None:
            return None, 0
        predictions = emotion_model.predict(processed_image, verbose=0)
        emotion_index = np.argmax(predictions[0])
        confidence = predictions[0][emotion_index]
        emotion_label = EMOTIONS[emotion_index]
        return emotion_label, float(confidence)
    except Exception as e:
        print(f"Error predicting emotion: {e}")
        return None, 0

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')  # Make sure this file is index.html or change this to index.htm

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
    
    # This HTML response is a bit unconventional but works.
    # A better way is using render_template('result.html', ...)
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
    # init_db() was moved to the top
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) # Added debug=True for local testing