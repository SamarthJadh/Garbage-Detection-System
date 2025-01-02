import cv2
from ultralytics import YOLO
import time
from flask import Flask, Response, render_template
from twilio.rest import Client
from dotenv import load_dotenv
import os
import random

# Load environment variables from the .env file
load_dotenv('new.env')

# Initialize the Flask app
app = Flask(__name__)

# Load Twilio credentials from the environment
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = 'whatsapp:+14155238886'  # Twilio WhatsApp Sandbox Number
DESTINATION_PHONE_NUMBER = 'whatsapp:+919579731091'  # Your WhatsApp number for notifications

# Initialize the Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Load the pre-trained and custom YOLO models
pretrained_model = YOLO('yolov8n.pt')  # Replace with the path to the pre-trained weights
custom_model = YOLO(r'best.pt')  # Replace with your custom weights

# Initialize the USB camera (0 is typically the default camera)
cap = cv2.VideoCapture(0)  # Replace 0 with the correct camera index if needed

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# List of specific objects to monitor for warnings
objects_to_warn = ['plastic Container', 'plastic bag or wrapper', 'cardboard boxes and cartons', 'paper', 'bottle']
detected_objects = {}
notified_objects = {}  # Track which objects have already triggered notifications

# Define a function to send WhatsApp alerts via Twilio
def send_whatsapp_alert(message):
    """Send a WhatsApp alert via Twilio"""
    try:
        msg = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=DESTINATION_PHONE_NUMBER
        )
        print(f"WhatsApp alert sent: {msg.sid}")
    except Exception as e:
        print(f"Failed to send WhatsApp alert: {e}")

# Function to get a random threshold
def get_random_threshold():
    return random.randint(3, 5)  # Random threshold between 3 and 7 seconds

# Define a function to annotate the frame
def annotate_frame(frame, boxes, model, detected_objects, min_confidence=0.5):
    for box in boxes:
        # Extract bounding box coordinates, confidence, and class
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = box.conf.item()
        class_id = int(box.cls.item())  # Get the class ID (index)
        class_name = model.names[class_id]  # Map class ID to class name using the model's names list

        if confidence > min_confidence:
            # Add object detection time tracking
            if class_name in detected_objects:
                detected_objects[class_name]['last_detected'] = time.time()
            else:
                detected_objects[class_name] = {
                    'first_detected': time.time(),
                    'last_detected': time.time(),
                    'threshold': get_random_threshold()  # Assign a random threshold
                }

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put class name and confidence label on the box
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Function to capture and stream frames for Flask
def generate_frames():
    global detected_objects, notified_objects
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference using both models
        results_pretrained = pretrained_model.predict(source=frame, conf=0.3, save=False, show=False)
        results_custom = custom_model.predict(source=frame, conf=0.1, save=False, show=False)

        # Filter detections based on confidence
        filtered_pretrained = [box for box in results_pretrained[0].boxes if box.conf.item() > 0.8]
        filtered_custom = [box for box in results_custom[0].boxes if box.conf.item() > 0.5]

        # Annotate the frame with detections and update tracking
        frame = annotate_frame(frame, filtered_pretrained, pretrained_model, detected_objects)
        frame = annotate_frame(frame, filtered_custom, custom_model, detected_objects)

        # Check for warnings
        current_time = time.time()
        for obj_name in objects_to_warn:
            if obj_name in detected_objects:
                obj_threshold = detected_objects[obj_name]['threshold']
                time_since_first_detected = current_time - detected_objects[obj_name]['first_detected']
                time_since_last_detected = current_time - detected_objects[obj_name]['last_detected']

                # Ensure the object has been consistently detected for the threshold time
                if time_since_first_detected > obj_threshold and obj_name not in notified_objects:
                    warning_message = f"Garbage detected: {obj_name} (Threshold: {obj_threshold}s)"
                    print(warning_message)
                    send_whatsapp_alert(warning_message)
                    notified_objects[obj_name] = True  # Mark object as notified
                    cv2.putText(frame, warning_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Remove object from notified if no longer detected
            elif obj_name in notified_objects and obj_name not in detected_objects:
                del notified_objects[obj_name]

        # Encode the frame as JPEG and return it as a byte stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Route to serve the video feed
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Main route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
