# ai_depth_object_app

A new Flutter project.

## Getting Started

This project is a starting point for a Flutter application.

A few resources to get you started if this is your first Flutter project:

- [Lab: Write your first Flutter app](https://docs.flutter.dev/get-started/codelab)
- [Cookbook: Useful Flutter samples](https://docs.flutter.dev/cookbook)

For help getting started with Flutter development, view the
[online documentation](https://docs.flutter.dev/), which offers tutorials,
samples, guidance on mobile development, and a full API reference.
# vision_buddy


# Vision Buddy (AI Blind Assist) 👁️🤖

**Vision Buddy** is a Senior Project designed as a smart haptic navigation application for the visually impaired. Built with Flutter, it leverages on-device AI (TensorFlow Lite) for real-time object detection and monocular depth estimation. The app connects seamlessly via Bluetooth Low Energy (BLE) to custom ESP32 haptic wristbands to provide directional vibration feedback, alongside voice alerts via Text-to-Speech (TTS).

---

## ✨ Key Features

* **Real-time Object Detection:** Identifies obstacles and everyday objects using a custom TFLite model.
* **Monocular Depth Estimation:** Calculates the distance of objects and general obstacles (Blobs) in the range of 1.5 - 2.5 meters.
* **Haptic Navigation (BLE):** Auto-connects to ESP32 microcontrollers (`Vibe_Left` and `Vibe_Right`) to send vibration commands based on obstacle direction and distance.
* **Thai Text-to-Speech (TTS):** Provides audible alerts and distance readouts in Thai ("ระวัง! คน ด้านหน้า", "พบ โต๊ะ ระยะ 2.0 เมตร ซ้าย").
* **Eco Mode:** A gesture-controlled black-screen mode to save battery while the AI and sensors continue running in the background.

---

## 🛠️ Tech Stack & Dependencies

* **Framework:** Flutter (Dart)
* **AI / ML:** `tflite_flutter` (NNAPI/GPU acceleration support)
* **Vision:** `camera`, `image`
* **Hardware Integration:** `flutter_blue_plus` (BLE), `permission_handler`
* **Audio:** `flutter_tts`

---

## 📂 Project Structure & Assets Setup

Before running the application, ensure that your Machine Learning models are placed correctly in the `assets` directory. The application strictly requires the following files:

```text
vision_buddy/
├── assets/
│   └── models/
│       ├── 1.tflite              # Object Detection Model (e.g., YOLO format)
│       ├── model_opt.tflite      # Monocular Depth Estimation Model
│       └── labels.txt            # Detection Class Labels
├── lib/
│   └── main.dart                 # Main Application Code
└── pubspec.yaml