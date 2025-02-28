from flask import Flask, render_template, redirect, url_for, request, flash, session, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import time
import threading
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import uuid
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size

# Get the absolute path to the application directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Create necessary folders with absolute paths
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
    folder_path = os.path.join(BASE_DIR, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

db = SQLAlchemy(app)

# Store processing status
processing_tasks = {}


# Task cleanup timer (runs every hour)
def cleanup_old_tasks():
    current_time = datetime.now()
    tasks_to_remove = []

    for task_id, task in processing_tasks.items():
        # Convert string time to datetime
        uploaded_at = datetime.strptime(task['uploaded_at'], '%Y-%m-%d %H:%M:%S')
        # Remove tasks older than 24 hours
        if current_time - uploaded_at > timedelta(hours=24):
            tasks_to_remove.append(task_id)

    for task_id in tasks_to_remove:
        del processing_tasks[task_id]

    # Schedule next cleanup
    threading.Timer(3600, cleanup_old_tasks).start()


# Start the cleanup task
cleanup_timer = threading.Timer(3600, cleanup_old_tasks)
cleanup_timer.daemon = True
cleanup_timer.start()


# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(10), nullable=False)  # 'admin' or 'user'


# Blog Model
class Blog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    author = db.Column(db.String(100), nullable=False)


# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Function to determine if file is an image or video
def is_video(filename):
    video_extensions = {'mp4', 'avi', 'mov'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in video_extensions


# Load deepfake detection model and face detector
def load_detection_model():
    try:
        # Model paths with absolute references
        model_path = os.path.join(BASE_DIR, 'deepfake-detection-model1.h5')
        prototxt_path = os.path.join(BASE_DIR, "deploy.prototxt")
        caffemodel_path = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

        # Check if deepfake model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Deepfake detection model not found at {model_path}")

        model = load_model(model_path, compile=False)

        # Using OpenCV's DNN face detector (more accurate than Haar cascades)
        if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path):
            detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
            detector_type = "dnn"
        else:
            # Fallback to Haar cascades
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            detector = cv2.CascadeClassifier(cascade_path)
            detector_type = "haar"

        return model, detector, detector_type

    except Exception as e:
        print(f"Error loading detection model: {e}")
        # Return None values to allow graceful handling elsewhere
        return None, None, None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already exists. Please choose another.", "danger")
            return redirect(url_for('signup'))

        # Hash password and create user
        hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_pw, role='user')
        db.session.add(new_user)
        db.session.commit()

        flash("Account created successfully! Please login.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['role'] = user.role
            flash("Login successful!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password.", "danger")

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('home'))


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    blogs = Blog.query.all()
    return render_template('dashboard.html', blogs=blogs, role=session['role'])


@app.route('/add_blog', methods=['GET', 'POST'])
def add_blog():
    if 'user_id' not in session or session['role'] != 'admin':
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        author = session['role']
        new_blog = Blog(title=title, content=content, author=author)
        db.session.add(new_blog)
        db.session.commit()
        flash("Blog added successfully!", "success")
        return redirect(url_for('dashboard'))
    return render_template('add_blog.html')


def detect_faces(frame, detector, detector_type):
    if detector_type == "dnn":
        # Using DNN face detector
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        detector.setInput(blob)
        detections = detector.forward()

        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                # Ensure coordinates are within the frame
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                faces.append((x1, y1, x2, y2, confidence))
    else:
        # Using Haar cascade face detector
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        faces = []
        for (x, y, w, h) in face_rects:
            faces.append((x, y, x + w, y + h, 1.0))  # 1.0 as placeholder confidence

    return faces


def process_image(task_id, filename):
    try:
        # Update status to processing
        processing_tasks[task_id]['status'] = 'processing'
        processing_tasks[task_id]['progress'] = 10

        # Load models
        model, detector, detector_type = load_detection_model()

        if model is None or detector is None:
            raise Exception("Failed to load detection models")

        # Read the image
        img_path = os.path.join(BASE_DIR, app.config['UPLOAD_FOLDER'], filename)
        frame = cv2.imread(img_path)

        if frame is None:
            raise Exception("Failed to read image file")

        processing_tasks[task_id]['progress'] = 30

        # Detect faces
        faces = detect_faces(frame, detector, detector_type)

        processing_tasks[task_id]['progress'] = 50

        detected = False
        results = []

        for face in faces:
            x1, y1, x2, y2, confidence = face

            # Ensure cropped image is valid
            if x1 >= x2 or y1 >= y2:
                continue

            # Crop and preprocess the face
            crop_img = frame[y1:y2, x1:x2]
            if crop_img.size == 0:
                continue

            data = cv2.resize(crop_img, (128, 128)) / 255.0
            data = data.reshape(-1, 128, 128, 3)

            processing_tasks[task_id]['progress'] = 70

            # Predict deepfake or real
            prediction = model.predict(data)[0][0]
            predicted_class = "DeepFake" if prediction >= 0.5 else "Real"
            color = (0, 0, 255) if prediction >= 0.5 else (0, 255, 0)

            # Store results for display
            results.append({
                'class': predicted_class,
                'confidence': float(prediction),
                'position': [int(x1), int(y1), int(x2), int(y2)]
            })

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{predicted_class} ({prediction:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            detected = True

        processing_tasks[task_id]['progress'] = 90

        # If no face detected, add message on image
        if not detected:
            h, w = frame.shape[:2]
            cv2.putText(frame, "No face detected", (w // 4, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            results.append({
                'class': 'No face detected',
                'confidence': 0,
                'position': []
            })

        # Save the result
        output_filename = f"result_{os.path.basename(filename)}"
        output_path = os.path.join(BASE_DIR, app.config['OUTPUT_FOLDER'], output_filename)
        cv2.imwrite(output_path, frame)

        processing_tasks[task_id]['progress'] = 100
        processing_tasks[task_id]['status'] = 'completed'
        processing_tasks[task_id]['result_file'] = output_filename
        processing_tasks[task_id]['results'] = results
        processing_tasks[task_id]['is_video'] = False

    except Exception as e:
        processing_tasks[task_id]['status'] = 'failed'
        processing_tasks[task_id]['error'] = str(e)
        print(f"Error processing image: {e}")


def process_video(task_id, filename):
    try:
        # Update status to processing
        processing_tasks[task_id]['status'] = 'processing'
        processing_tasks[task_id]['progress'] = 5

        # Load models
        model, detector, detector_type = load_detection_model()

        if model is None or detector is None:
            raise Exception("Failed to load detection models")

        # Load video
        video_path = os.path.join(BASE_DIR, app.config['UPLOAD_FOLDER'], filename)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise Exception("Could not open video file")

        # Get video properties
        frameRate = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output video settings
        output_filename = f"result_{os.path.basename(filename)}"
        output_path = os.path.join(BASE_DIR, app.config['OUTPUT_FOLDER'], output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_fps = 10  # Reduced FPS for smoother playback
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (frame_width, frame_height))

        frame_id = 0
        saved_frames = []
        results = []
        max_frames_to_process = min(500, total_frames)  # Limit processing to 500 frames max
        skip_frames = max(1, int(total_frames / max_frames_to_process))

        processing_tasks[task_id]['progress'] = 10

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Process every N frames to speed up processing
            if frame_id % skip_frames == 0:
                faces = detect_faces(frame, detector, detector_type)

                frame_results = []
                for face in faces:
                    x1, y1, x2, y2, confidence = face

                    # Ensure cropped image is valid
                    if x1 >= x2 or y1 >= y2:
                        continue

                    # Crop and preprocess the face
                    crop_img = frame[y1:y2, x1:x2]
                    if crop_img.size == 0:
                        continue

                    data = cv2.resize(crop_img, (128, 128)) / 255.0
                    data = data.reshape(-1, 128, 128, 3)

                    # Predict deepfake or real
                    prediction = model.predict(data)[0][0]
                    predicted_class = "DeepFake" if prediction >= 0.5 else "Real"
                    color = (0, 0, 255) if prediction >= 0.5 else (0, 255, 0)

                    # Store frame result
                    frame_results.append({
                        'class': predicted_class,
                        'confidence': float(prediction),
                        'position': [int(x1), int(y1), int(x2), int(y2)],
                        'frame': frame_id
                    })

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{predicted_class} ({prediction:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Add frame to output if we found faces
                if frame_results:
                    saved_frames.append(frame.copy())
                    results.extend(frame_results)

            frame_id += 1

            # Update progress periodically
            if frame_id % 20 == 0:
                progress = min(90, 10 + int(80 * frame_id / total_frames))
                processing_tasks[task_id]['progress'] = progress

            # Limit processing time
            if frame_id > max_frames_to_process:
                break

        # Write detected frames to output video
        for detected_frame in saved_frames:
            out.write(detected_frame)

        # If no frames were saved, add a "No faces detected" frame
        if not saved_frames:
            blank_frame = np.zeros((frame_height, frame_width, 3), np.uint8)
            cv2.putText(blank_frame, "No faces detected in video", (frame_width // 4, frame_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(blank_frame)

        # Release resources
        cap.release()
        out.release()

        processing_tasks[task_id]['progress'] = 100
        processing_tasks[task_id]['status'] = 'completed'
        processing_tasks[task_id]['result_file'] = output_filename
        processing_tasks[task_id]['results'] = results
        processing_tasks[task_id]['is_video'] = True

    except Exception as e:
        processing_tasks[task_id]['status'] = 'failed'
        processing_tasks[task_id]['error'] = str(e)
        print(f"Error processing video: {e}")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file uploaded!", "danger")
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash("No selected file!", "danger")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Check file size
            file_data = file.read()
            file_size = len(file_data)

            if file_size > app.config['MAX_CONTENT_LENGTH']:
                flash(f"File too large! Maximum size is {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)}MB",
                      "danger")
                return redirect(request.url)

            # Reset file pointer
            file.seek(0)

            # Generate unique task ID
            task_id = str(uuid.uuid4())

            # Secure the filename
            filename = secure_filename(file.filename)
            file_path = os.path.join(BASE_DIR, app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Initialize task status
            processing_tasks[task_id] = {
                'status': 'queued',
                'progress': 0,
                'filename': filename,
                'uploaded_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            # Start processing in a background thread
            if is_video(filename):
                thread = threading.Thread(target=process_video, args=(task_id, filename))
            else:
                thread = threading.Thread(target=process_image, args=(task_id, filename))

            thread.daemon = True
            thread.start()

            # Redirect to status page
            return redirect(url_for('status_page', task_id=task_id))
        else:
            flash("File type not allowed! Please upload images (jpg, png) or videos (mp4, avi, mov).", "danger")
            return redirect(request.url)

    return render_template('predict.html')


@app.route('/status_page/<task_id>')
def status_page(task_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if task_id not in processing_tasks:
        flash("Task not found!", "danger")
        return redirect(url_for('predict'))

    return render_template('predict.html', task_id=task_id)


@app.route('/api/status/<task_id>', methods=['GET'])
def check_status(task_id):
    if task_id not in processing_tasks:
        return jsonify({'success': False, 'error': 'Task not found'}), 404

    task = processing_tasks[task_id]
    if task['status'] == 'completed':
        response = {
            'success': True,
            'status': 'completed',
            'progress': 100,
            'output_url': url_for('send_output_file', filename=task['result_file']),
            'download_url': url_for('download_result', task_id=task_id),
            'results': task.get('results', []),
            'is_video': task.get('is_video', False)
        }
        return jsonify(response)
    elif task['status'] == 'failed':
        return jsonify({
            'success': False,
            'status': 'failed',
            'error': task.get('error', 'Processing failed')
        })
    else:
        return jsonify({
            'success': True,
            'status': task['status'],
            'progress': task.get('progress', 0)
        })


@app.route('/download/<task_id>')
def download_result(task_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if task_id not in processing_tasks or processing_tasks[task_id]['status'] != 'completed':
        flash("Result not available!", "danger")
        return redirect(url_for('predict'))

    result_file = processing_tasks[task_id]['result_file']
    return send_file(os.path.join(BASE_DIR, app.config['OUTPUT_FOLDER'], result_file), as_attachment=True)


@app.route('/outputs/<filename>')
def send_output_file(filename):
    return send_file(os.path.join(BASE_DIR, app.config['OUTPUT_FOLDER'], filename))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()

        # Ensure the admin user exists - use environment variables in production
        admin_username = os.environ.get('ADMIN_USERNAME', 'Admin@123')
        admin_password = os.environ.get('ADMIN_PASSWORD', 'Admin@123')

        admin_user = User.query.filter_by(username=admin_username).first()
        if not admin_user:
            hashed_pw = generate_password_hash(admin_password, method='pbkdf2:sha256')
            admin = User(username=admin_username, password=hashed_pw, role="admin")
            db.session.add(admin)
            db.session.commit()
            print("Admin user has been created!")

    app.run(debug=True)