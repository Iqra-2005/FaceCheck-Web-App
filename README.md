# 🌐 FaceCheck – AI-Based Smart Attendance System

A web-based attendance system using facial recognition. The system allows users to register their face (multiple frames per person), stores embeddings, and recognises faces in real-time to mark attendance automatically, works on multiple devices and asures user security by avoiding the storage of user images.


## 📝 Features

- **Face Registration**: Capture multiple frames of a person’s face, generate embeddings, and store them along with metadata (name, roll number, class).

- **Face Recognition**: Real-time detection and matching of a face against saved embeddings for attendance marking.

- **Web Interface**: Built using Flask (Python) with front-end HTML/CSS templates.

- **Database**: Stores user info, embeddings, and attendance logs using SQLite.

- **Extensible**: Can integrate a custom embedding model (e.g., Siamese network) for improved accuracy.

---
## 🛠 System Overview

 **1)Registration Phase**

  - User enters name, roll number, class.    
 
  - Webcam captures multiple face frames.    
 
  - Each frame is converted into an embedding vector.   
 
  - Embeddings + user info are stored in the database.

**2)Recognition Phase**

- Webcam captures frames continuously.

- Detect face(s) in real time.

- Generate embeddings for detected faces.

- Compare with stored embeddings (e.g., cosine similarity).

- If matched → attendance is marked with timestamp.

- User feedback shown on screen (names are shown if recognised).

**3)Database / Storage**

- User metadata table (name, roll no., class).

- Embeddings table (multiple embeddings per user).

- Attendance logs (user ID, timestamp, class/session).

**4)Web App Architecture**

- app.py — Main Flask routes (/register, /recognise, /attendance, etc.)

- db_utils.py — DB helper functions (insert, read, update).

- templates/ — HTML pages for UI (registration, recognition, attendance).

- static/ — CSS/JS files.

- face_attendance.db — SQLite database.

- requirements.txt — Python dependencies.

---

## ✅ Getting Started
**1)Prerequisites**

- Python 3.x

- Webcam

- Required packages: Flask, OpenCV, TensorFlow/Keras, NumPy, SQLite3, etc.

**2)Setup**

i) Clone the repository 
    
      git clone https://github.com/Iqra-2005/FaceCheck-Web-App.git
    
      cd FaceCheck-Web-App
    
ii) Create & activate virtual environment (optional)
  
      python3 -m venv venv
      
      source venv/bin/activate     # Linux/macOS
      
      venv\Scripts\activate        # Windows
      
iii) Install dependencies 

    pip install -r requirements.txt
  
iv) Add your custom embedding model
  
    Place embedding_model.h5 in the project folder and update the model path in the code.
      
    Run the App : python app.py

Open in browser: http://127.0.0.1:5000/

---

## 📁 Project Structure

    FaceCheck-Web-App/
    │
    ├─ app.py                # Main Flask application
    ├─ db_utils.py           # Database utilities
    ├─ requirements.txt      # Python dependencies
    ├─ face_attendance.db    # SQLite database
    │_embedding_model.H5     # Siamese Model for embedding generation (not present in the current repo)
    |
    ├─ templates/            # HTML templates
    │   ├─ register.html
    │   ├─ recognise.html
    │   ├─ attendance.html
    │   └─ ...
    │
    ├─ static/               # CSS, JS, images
    │   └─ ...
    │_

---

## 🎯 How to Use

**1) Register a New User**

- Open the registration page.

- Enter name, roll number, class.

- Click Start Capture to record multiple face images.

- Embeddings are generated and stored automatically.

**2) Recognise / Mark Attendance**

- Open the recognition page.

- Allow webcam access.

- The system detects the face and compares it with saved embeddings.

- If matched → attendance is marked with timestamp.

**3) View Attendance**

- Go to the attendance logs page to view all records.

- You can filter attendance based on date and subject and view past records

---

## 📝 Notes

- Created as an academic project for TY BSc Data Science & AI (2025–2026).

- Works with both default and custom (Siamese-based) embedding models.

- No open-source license has been added. You must request permission before using this code.
