#  real time image processing from cctv camera
# Install OpenCV inside Google Colab
!pip install opencv-python
# Now import OpenCV
import cv2
# Let's check OpenCV version
print("OpenCV Version:", cv2.__version__)

from google.colab import files
uploaded = files.upload()

import cv2
import matplotlib.pyplot as plt

# Load the video
video_path = 'sample_traffic_video.mp4.mp4'  # Your uploaded video file name
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()  # Read each frame
    if not ret:
        break  # Exit loop if video is over

    # Show the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.show()
    break  # Show only the first frame for testing

cap.release()

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap='gray')
    plt.axis('off')
    plt.show()

cap.release()

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Upload your video manually
from google.colab import files
uploaded = files.upload()

model = YOLO('yolov8n.pt')  # 'n' = Nano version (small and fast)

 Open the uploaded video
video_path = list(uploaded.keys())[0]  # Get uploaded filename
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 prediction
    results = model.predict(frame, imgsz=640, conf=0.5)

    # Get the annotated frame
    annotated_frame = results[0].plot()

    # Display
    plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

cap.release()

https://colab.research.google.com/drive/1sbASaGFFSuaI3W_Wd3rfpItXeCqLZNYJ?authuser=1#scrollTo=Hdix0u0d5qa_

# machine learning model ai foe accidet prevention and prediction
# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load historical data
data = pd.read_csv('historical_accident_data.csv')

# Feature columns
feature_columns = ['latitude', 'longitude', 'traffic_density', 'vibration_level', 'pressure_value', 'hour_of_day', 'day_of_week', 'weather_condition']
target_column = 'accident_risk'  # 1 = High risk, 0 = Low risk

X = data[feature_columns]
y = data[target_column]

# Split data into Train (60%), Validation (20%), Test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Save split data
X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
y_val.to_csv('y_val.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'accident_traffic_predictor.pkl')

print("\n✅ Training complete. Model saved as 'accident_traffic_predictor.pkl'")
print("✅ Training, validation, and test datasets saved separately.")

# validate_model.py

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load validation data
X_val = pd.read_csv('X_val.csv')
y_val = pd.read_csv('y_val.csv')

# Load trained model
model = joblib.load('accident_traffic_predictor.pkl')

# Predict on validation set
val_predictions = model.predict(X_val)

# Evaluate performance
print("\n🔵 Validation Results:")
print("Accuracy:", accuracy_score(y_val, val_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_val, val_predictions))
print("Classification Report:\n", classification_report(y_val, val_predictions))

# test_model.py

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load test data
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

# Load trained model
model = joblib.load('accident_traffic_predictor.pkl')

# Predict on test set
test_predictions = model.predict(X_test)

# Evaluate performance
print("\n🟠 Test Results:")
print("Accuracy:", accuracy_score(y_test, test_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, test_predictions))
print("Classification Report:\n", classification_report(y_test, test_predictions))

{
  "latitude": 12.9716,
  "longitude": 77.5946,
  "traffic_density": 85,
  "vibration_level": 0.25,
  "pressure_value": 101200,
  "hour_of_day": 18,
  "day_of_week": 3,
  "weather_condition": 1
}
# real_time_predict.py

import numpy as np
import joblib

# Load trained model
model = joblib.load('accident_traffic_predictor.pkl')

# Real-time input (from sensor / API)
real_time_input = np.array([[12.9716, 77.5946, 85, 0.25, 101200, 18, 3, 1]])  # shape (1,8)

# Predict
prediction = model.predict(real_time_input)

if prediction[0] == 1:
    print("⚠ High Accident/Traffic Risk! Immediate Action Needed.")
else:
    print("✅ Normal Conditions.")


# trafic managemnt using RFID
import time
import random

# Simulate RFID tag IDs
rfid_tags = ['VEH123', 'VEH456', 'VEH789', 'VEH101']

def detect_vehicle():
    # Randomly simulate a vehicle detection
    vehicle_detected = random.choice(rfid_tags)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return vehicle_detected, timestamp

# Main loop simulating RFID reader
for _ in range(10):  # Simulate 10 reads
    vehicle_id, timestamp = detect_vehicle()
    print(f"Detected vehicle: {vehicle_id} at {timestamp}")
    time.sleep(random.uniform(1, 3))  # Wait 1 to 3 seconds between reads
CREATE TABLE rfid_detections (
    id SERIAL PRIMARY KEY,
    vehicle_id VARCHAR(20),
    detection_point VARCHAR(50),
    timestamp TIMESTAMP
);


# GPS
import time
import random

# Simulate GPS coordinates around a city (latitude, longitude)
def generate_gps_data():
    lat = 12.9716 + random.uniform(-0.005, 0.005)
    lon = 77.5946 + random.uniform(-0.005, 0.005)
    speed = random.uniform(0, 60)  # Speed in km/h
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return lat, lon, speed, timestamp

# Simulate real-time GPS tracking
for _ in range(10):  # 10 data points
    latitude, longitude, speed, timestamp = generate_gps_data()
    print(f"GPS: {latitude}, {longitude} | Speed: {speed:.2f} km/h at {timestamp}")
    time.sleep(2)  # update every 2 seconds

  CREATE TABLE gps_tracking (
    id SERIAL PRIMARY KEY,
    vehicle_id VARCHAR(20),
    latitude FLOAT,
    longitude FLOAT,
    speed FLOAT,
    timestamp TIMESTAMP
);

# storing edge data base
import sqlite3

# Connect to a local SQLite database (creates file if not exists)
conn = sqlite3.connect('traffic_edge.db')
cursor = conn.cursor()

# Create table for analyzed data
cursor.execute('''
CREATE TABLE IF NOT EXISTS traffic_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    latitude REAL,
    longitude REAL,
    traffic_status TEXT,
    accident_detected BOOLEAN,
    vehicle_count INTEGER,
    timestamp TEXT
)
''')

conn.commit()
conn.close()

print("Database and table created successfully.")

# FastAPI Server to Accept and Serve Data
# filename: server.py
from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
from typing import List

app = FastAPI()

# Pydantic model for data input
class TrafficData(BaseModel):
    latitude: float
    longitude: float
    traffic_status: str
    accident_detected: bool
    vehicle_count: int
    timestamp: str

# Save data into database
@app.post("/add_data/")
def add_data(data: TrafficData):
    conn = sqlite3.connect('traffic_edge.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO traffic_data (latitude, longitude, traffic_status, accident_detected, vehicle_count, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (data.latitude, data.longitude, data.traffic_status, data.accident_detected, data.vehicle_count, data.timestamp))
    conn.commit()
    conn.close()
    return {"message": "Data inserted successfully"}

# Get all data
@app.get("/get_data/")
def get_data():
    conn = sqlite3.connect('traffic_edge.db')
    cursor = conn.cursor()
    cursor.execute('SELECT latitude, longitude, traffic_status, accident_detected, vehicle_count, timestamp FROM traffic_data')
    rows = cursor.fetchall()
    conn.close()

    return [{"latitude": row[0], "longitude": row[1], "traffic_status": row[2], "accident_detected": row[3], "vehicle_count": row[4], "timestamp": row[5]} for row in rows]

  #Connect Your App to this Server
  import 'package:http/http.dart' as http;
import 'dart:convert';

Future<void> sendTrafficData() async {
  final url = Uri.parse('http://192.168.1.100:8000/add_data/'); // IP of your edge device

  final response = await http.post(url,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        "latitude": 12.9716,
        "longitude": 77.5946,
        "traffic_status": "Heavy",
        "accident_detected": true,
        "vehicle_count": 45,
        "timestamp": "2025-04-28T15:00:00Z"
      }));

  if (response.statusCode == 200) {
    print('Data sent successfully');
  } else {
    print('Failed to send data');
  }
}

#Future<List<dynamic>> fetchTrafficData() async {
  final url = Uri.parse('http://192.168.1.100:8000/get_data/');
  final response = await http.get(url);

  if (response.statusCode == 200) {
    return jsonDecode(response.body);
  } else {
    throw Exception('Failed to load data');
  }
}
Future<List<dynamic>> fetchTrafficData() async {
  final url = Uri.parse('http://192.168.1.100:8000/get_data/');
  final response = await http.get(url);

  if (response.statusCode == 200) {
    return jsonDecode(response.body);
  } else {
    throw Exception('Failed to load data');
  }
}
