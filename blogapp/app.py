from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import sqlite3
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from skimage.feature import local_binary_pattern
import pywt
import re

# Load model
model = load_model("model2.keras")
class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# Flask setup
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# SQLite helper
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

# Create users table if not exists
with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()

# 🧠 Handcrafted Feature Extractor
def extract_handcrafted_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {}

    features = {}
    img = cv2.resize(img, (256, 256))
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    edges = cv2.Canny(blurred, 100, 200)
    features['Edge Count'] = int(np.sum(edges > 0))

    corners = cv2.cornerHarris(np.float32(blurred), 2, 3, 0.04)
    features['Corner Count'] = int(np.sum(corners > 0.01 * corners.max()))

    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    features['Ridge Thickness'] = round(np.sum(binary == 255) / (256 * 256), 3)
    features['Valley Thickness'] = round(np.sum(binary == 0) / (256 * 256), 3)

    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float") / (hist.sum() + 1e-6)
    for i, val in enumerate(hist[:5]):
        features[f'LBP_{i}'] = round(val, 4)

    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    features['Wavelet_cA_mean'] = round(np.mean(cA), 2)
    features['Wavelet_cH_std'] = round(np.std(cH), 2)
    features['Wavelet_cV_std'] = round(np.std(cV), 2)
    features['Wavelet_cD_std'] = round(np.std(cD), 2)

    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, ksize=3)
    angles = np.arctan2(gy, gx) * (180 / np.pi)
    features['Mean Orientation'] = round(np.mean(angles), 2)

    return features

@app.route('/')
def root():
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Email validation
        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        if not re.match(email_pattern, email):
            flash("Invalid email format.")
            return redirect(url_for('signup'))

        # Password validation
        if len(password) < 6:
            flash("Password must be at least 6 characters.")
            return redirect(url_for('signup'))

        if password != confirm_password:
            flash("Passwords do not match.")
            return redirect(url_for('signup'))

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, email))
        existing_user = cursor.fetchone()

        if existing_user:
            flash("Username or Email already exists.")
            conn.close()
            return redirect(url_for('signup'))

        cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                       (username, email, password))
        conn.commit()
        conn.close()

        flash("Signup successful. Please login.")
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session['user'] = username
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(64, 64))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = round(np.max(predictions) * 100, 2)

            handcrafted_features = extract_handcrafted_features(filepath)

            return render_template('result.html',
                                   predicted_class=predicted_class,
                                   confidence=confidence,
                                   image_url=url_for('static', filename=f'uploads/{filename}'),
                                   features=handcrafted_features)
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
