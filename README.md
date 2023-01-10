# Surgical Position Tracking 

This repository contains the Talke Biomedical Device Lab's surgical position tracking implementation using Python. 

![til](./marker_tracking.gif)

## Description of Files

### Scripts
1. **calibration.py** : Details the steps used in camera calibration using the folder **CalibrationImages/**. 

        python calibration.py

2. **marker_tracking.py** : Marker tracking implementation using the provided video **Ref4_S3.mp4**.

        python marker_tracking.py

### Other
3. **kalman.ipymb** : Post-processing Kalman filter implementation on generated csvs. **marker_tracking.py** provides an online implementation of Kalman filtering.

4. **Ref4_S3.mp4** : Provided video taken from an XYZ platform. 

5. **CalibrationImages/** : Provided calibration imges for **calibration.py**.

## Implementation

The script **marker_tracking.py** performs the following:

1. At each frame, locate the ArUco markers. 

2. Using the calibration files in **CalibrationImages/**, take the 3D coordinates of the center of each ArUco marker with respect to the camera. 

3. Using these 3D coordinates, perform online linear Kalman filtering.

4. Export the final file as a CSV with the XYZ coordinates for further processing and data visualization. 