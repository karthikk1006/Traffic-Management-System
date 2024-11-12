# Traffic Lane Image Processing and Analysis App

This Streamlit application processes traffic lane images, detects vehicles using a YOLOv8 model, calculates green signal times for each lane, and visualizes the results through images and graphs.

## Features
- **Upload Traffic Lane Images**: Supports up to 4 images, one per lane.
- **Vehicle Detection**: Counts vehicles in each lane using the YOLOv8 model.
- **Green Signal Calculation**: Calculates green signal durations based on vehicle counts.
- **Visualization**: Displays original and processed images with bounding boxes, along with bar, pie, and line plots for analysis.

## Requirements

- **Python 3.8 or higher**
- **Dependencies**:
  - `streamlit`
  - `torch`
  - `ultralytics`
  - `opencv-python`
  - `matplotlib`
  - `numpy`

## Installation

1. **Clone the repository** (if applicable) or download the code.
    ```bash
    git clone <repository-link>
    cd Traffic-Management-System
    ```

2. **Install dependencies**:
    Ensure you have Python 3.8 or higher installed, and then run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Streamlit app**:
    ```bash
    streamlit run main.py
    ```

2. **Upload Images**:
   - Use the sidebar to upload 4 images, each representing traffic in one of the lanes.
   - Images should be in `.jpg`, `.jpeg`, or `.png` format.

3. **Process and View Results**:
   - Once images are uploaded, the app will:
     - Detect and count vehicles in each lane.
     - Calculate green signal times based on vehicle counts.
     - Display original and processed images with bounding boxes.
     - Visualize data using:
       - **Bar Chart**: Vehicle count per lane.
       - **Pie Chart**: Green signal time distribution across lanes.
       - **Line Plot**: Relationship between vehicle count and green signal time.

## Deployed Link

You can access the app directly at [Traffic Management App](https://traffic-management-system-karthik.streamlit.app/)


