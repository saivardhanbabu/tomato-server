from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import base64

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained YOLOv8 model
model = YOLO('/Users/saivardhanbabugunda/Downloads/Yolov8-final.pt')  # Replace with the path to your trained model

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform prediction
        results = model(file_path)

        # Load the image using OpenCV
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency

        # Extract bounding boxes, class labels, and confidence scores
        predictions = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_names = result.names  # Class names dictionary

            for box, class_id, confidence in zip(boxes, class_ids, confidences):
                x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
                label = class_names[int(class_id)]

                # Draw bounding box on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                predictions.append({
                    'prediction': label,
                    'confidence': float(confidence),
                    'bounding_box': [x1, y1, x2, y2]
                })

        # Save the annotated image temporarily
        annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_' + filename)
        cv2.imwrite(annotated_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Convert the annotated image to base64
        with open(annotated_image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

        # Return the prediction results and the annotated image
        return jsonify({
            'predictions': predictions,
            'annotated_image': encoded_image
        })

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)