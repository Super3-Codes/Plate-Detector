import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
app = Flask(__name__)

model = YOLO('best.pt')

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"})

    # Read the image file
    image_file = request.files["image"]

    # Convert image file to numpy array
    nparr = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform object detection
    results = model.predict(image)
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()

    boxes = [[xmin, ymin, xmax, ymax] for xmin, ymin, xmax, ymax in boxes]
    classes = [names[class_idx] for class_idx in classes]
    confidences = [round(confidence, 2) for confidence in confidences]

    # Draw bounding boxes on the image
    for (xmin, ymin, xmax, ymax), class_name, confidence in zip(boxes, classes, confidences):
        # Draw bounding box
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

        # Add label
        label = f"{class_name} {confidence}"
        cv2.putText(image, label, (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Convert image to RGB (for displaying with matplotlib)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # Convert the image to bytes
    _, img_encoded = cv2.imencode(".jpg", image)
    img_bytes = img_encoded.tobytes()

    return img_bytes, 200, {"Content-Type": "image/jpeg"}




if __name__ == "__main__":
    app.run(debug=True )