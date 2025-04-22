from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
from collections import Counter

app = Flask(__name__)
CORS(app)  # Allow frontend to access backend

# Load the YOLO model once at the start
model = YOLO("best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image
        file = request.files['image']
        img = Image.open(file.stream)

        # Perform prediction with YOLO
        results = model.predict(img)

        # Get the bounding boxes and class names
        boxes = results[0].boxes.data
        class_names = [model.names[int(cls)] for cls in boxes[:, 5]]

        # Count occurrences of each class
        count = Counter(class_names)

        # Map the counts to specific blood cells
        response = {
            "WBC": count.get("WBC", 0),  # Replace with correct class names if needed
            "RBC": count.get("RBC", 0),
            "Platelets": count.get("platelets", 0)  # Adjust if the class name is different
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
