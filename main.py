import streamlit as st
from PIL import Image
import torch
import cv2
import os
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

# Initialize YOLO model
model = YOLO('yolov8n.pt')

# Function to count vehicles in an image
def count_vehicles(img):
    results = model(img)  # Run YOLO on the image
    vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']
    vehicle_count = 0
    
    for result in results:
        for cls, conf, (xmin, ymin, xmax, ymax) in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
            if result.names[int(cls)] in vehicle_classes:
                vehicle_count += 1
                
    return vehicle_count

# Function to draw bounding boxes
def draw_bounding_boxes(img):
    results = model(img)
    img_array = np.array(img)
    
    for result in results:
        for cls, conf, (xmin, ymin, xmax, ymax) in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
            if result.names[int(cls)] in ['car', 'motorcycle', 'bus', 'truck']:
                cv2.rectangle(img_array, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.putText(img_array, result.names[int(cls)], (int(xmin), int(ymin) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return Image.fromarray(img_array)

# Streamlit app setup
st.title("Traffic Lane Image Processing")

# Sidebar for image uploads
st.sidebar.header("Upload Lane Images")
lane_images = []
for i in range(1, 5):
    uploaded_image = st.sidebar.file_uploader(f"Upload Lane {i} Image", type=["jpg", "jpeg", "png"], key=f"lane_{i}")
    if uploaded_image is not None:
        lane_images.append(Image.open(uploaded_image))
    else:
        lane_images.append(None)

if all(lane_images):
    # Count vehicles and calculate green signal durations
    vehicle_counts = [count_vehicles(img) for img in lane_images]
    total_vehicles = sum(vehicle_counts) if sum(vehicle_counts) > 0 else 1  # Prevent division by zero
    max_green_time = 60  # Max green signal time in seconds
    green_times = [int((count / total_vehicles) * max_green_time) for count in vehicle_counts]

    # Display original and processed images
    st.write("### Original and Processed Images with Bounding Boxes")
    for i, img in enumerate(lane_images):
        st.markdown(
        f"""
        <div style="padding: 10px; border: 2px solid #4CAF50; border-radius: 10px; margin-bottom: 10px;">
            <h3 style="color: #4CAF50;">Lane {i + 1}</h3>
            <p style="font-size: 18px;"><strong>Vehicle Count:</strong> {vehicle_counts[i]}</p>
            <p style="font-size: 18px;"><strong>Green Signal Time:</strong> {green_times[i]} seconds</p>
        </div>
        """, unsafe_allow_html=True
    )
        
        col1, col2 = st.columns(2)
        col1.image(img, caption=f"Original Lane {i+1}", use_container_width=True)
        
        processed_img = draw_bounding_boxes(img)
        col2.image(processed_img, caption=f"Processed Lane {i+1}", use_container_width=True)

        # Analysis with graphs
    st.write("## Traffic Analysis")
    
    # Bar chart for vehicle counts
    st.write("### Vehicle Count per Lane")
    fig, ax = plt.subplots()
    ax.bar([f"Lane {i+1}" for i in range(len(vehicle_counts))], vehicle_counts, color="skyblue")
    ax.set_xlabel("Lane")
    ax.set_ylabel("Vehicle Count")
    ax.set_title("Vehicle Count in Each Lane")
    st.pyplot(fig)
    
    # Pie chart for green signal time distribution
    st.write("### Green Signal Time Distribution")
    fig, ax = plt.subplots()
    ax.pie(green_times, labels=[f"Lane {i+1}" for i in range(len(green_times))], autopct='%1.1f%%', startangle=90, colors=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"])
    ax.set_title("Green Signal Time Distribution Across Lanes")
    st.pyplot(fig)
    
    # Line plot for Vehicle Count vs. Green Signal Time
    st.write("### Vehicle Count vs. Green Signal Time")
    fig, ax = plt.subplots()
    ax.plot(vehicle_counts, green_times, marker='o', color="purple", linestyle='-')
    ax.set_xlabel("Vehicle Count")
    ax.set_ylabel("Green Signal Time (seconds)")
    ax.set_title("Vehicle Count vs. Green Signal Time")
    for i, txt in enumerate([f"Lane {i+1}" for i in range(len(vehicle_counts))]):
        ax.annotate(txt, (vehicle_counts[i], green_times[i]), textcoords="offset points", xytext=(5,5), ha='center')
    st.pyplot(fig)
else:
    st.sidebar.write("Please upload images for all lanes to proceed.")