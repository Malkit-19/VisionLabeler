# VisionLabeler

Transform Videos into YOLO Datasets Automatically

VisionLabeler is an AI-powered dashboard that converts videos into structured YOLO datasets through automatic object detection, frame extraction, and label generation. The platform enables users to upload videos, detect multiple object classes using YOLO models, and automatically generate images and corresponding annotation files for dataset creation.

## Features

* Automatic Object Detection using YOLO
* Multi-Class Detection Support

  * Pedestrian Detection
  * Rickshaw Detection
  * 2-Wheeler Detection
  * Car Detection
  * Speed Breaker Detection
* Automatic Frame Extraction
* Automatic YOLO Label Generation
* Custom YOLO Model (.pt) Upload Support
* Real-Time Detection Preview
* Detection Progress Monitoring
* Organized Dataset Generation
* Image and Label Consistency Validation

## How It Works

1. Upload a video.
2. Select one or more detection classes.
3. Optionally upload a custom YOLO model.
4. Run detection.
5. VisionLabeler automatically:

   * Detects objects
   * Extracts frames
   * Generates YOLO annotation files
   * Creates a structured dataset directory

## Technologies Used

* Python
* Flask
* OpenCV
* Ultralytics YOLO
* HTML
* CSS
* JavaScript

## Project Structure

VisionLabeler/
├── final_app.py
├── requirements.txt
├── README.md
├── database/
├── uploads/
├── outputs/
└── website/

## Future Enhancements

* Lane Detection Integration
* Dataset Export Options
* Cloud Deployment
* Advanced Analytics Dashboard
* Model Management System

## Author

Malkit Singh Salas

Summer Internship 2026 Project
