
"""
Facial Emotion Recognition Web Application
Flask app that detects emotions from uploaded images
"""

import os
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

app = Flask(name)
app.secret_key = 'your_secret_key_here_change_in_production'  # For flash messages

# File upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Emotion labels (must match training order)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Emotion responses (what to tell the user)
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

def init_db():
    """Initialize the database and create tables if they don't exist"""
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
        model = keras.models.load_model('face_emotionModel.h5')
        print("✓ Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load model once when app starts
emotion_model = load_emotion_model()

# ============================================================================
# IMAGE PROCESSING & PREDICTION
# ============================================================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """
    Preprocess image for emotion detection
    - Convert to grayscale
    - Resize to 48x48
    - Normalize pixel values
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to 48x48 (model input size)
        resized = cv2.resize(gray, (48, 48))
        
        # Normalize pixel values to 0-1
        normalized = resized / 255.0
        
        # Reshape for model input (1, 48, 48, 1)
        # 1 = batch size, 48x48 = image dimensions, 1 = grayscale channel
        final = normalized.reshape(1, 48, 48, 1)
        
        return final
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_emotion(image_path):
    """
    Predict emotion from an image
    Returns: (emotion_label, confidence)
    """
    if emotion_model is None:
        return None, 0
    
    try:
        # Preprocess image
        processed_image = preprocess_image(image_path)
        
        if processed_image is None:
            return None, 0
        
        # Make prediction
        predictions = emotion_model.predict(processed_image, verbose=0)
        
        # Get emotion with highest probability
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
    """Home page with the form"""
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    """Handle form submission"""
    
    # Get form data
    name = request.form.get('name', '').strip()
    email = request.form.get('email', '').strip()
    age = request.form.get('age', '').strip()
    
    # Validate form data
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
    
    # Check if image was uploaded
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
    
    # Save uploaded image
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    
    # Predict emotion
    emotion, confidence = predict_emotion(filepath)
    
    if emotion is None:
        flash('Error processing image. Please try again.', 'error')
        return redirect(url_for('index'))
    
    # Save to database
    success = save_to_database(name, email, age, emotion, filepath)
    
    if not success:
        flash('Error saving data. Please try again.', 'error')
        return redirect(url_for('index'))
    
    # Get emotion response
    response_message = EMOTION_RESPONSES.get(emotion, "Emotion detected!")
    
    # Create result message
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
# MAIN
# ============================================================================

if name == 'main':
    # Initialize database on startup
    init_db()
    
    # Check if model exists
    if emotion_model is None:
        print("\n" + "="*60)
        print("WARNING: Model file 'face_emotionModel.h5' not found!")
        print("Please train the model first using model_training.py")
        print("="*60 + "\n")
    
    # Run the app
    print("\n" + "="*60)
    print("Starting Facial Emotion Recognition Web App...")
    print("Visit: http://127.0.0.1:5000")
    print("Press CTRL+C to stop the server")
    print("="*60 + "\n")
    
    # Get port from environment variable (for deployment) or use 5000
    import os
    port = int(os.environ.get('PORT', 5000))
    
    app.run(debug=False, host='0.0.0.0', port=port)
# Note: Set debug=True for development, but False for production
