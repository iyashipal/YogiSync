# YogiSync 🧘‍♀️

[![BrickHack 11](https://img.shields.io/badge/BrickHack-11-orange)](https://devpost.com/software/yogisync)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-red.svg)](https://mediapipe.dev/)

A Smart Yoga App that uses AI to provide real-time feedback and guidance for anyone looking to learn yoga or perfect their form.

**🏆 Submitted to BrickHack 11**

[View on Devpost](https://devpost.com/software/yogisync)


## 📖 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Team](#team)


## 🎯 Overview

YogiSync leverages cutting-edge AI and computer vision technologies to create an all-purpose yoga guidance solution. Whether you're a beginner looking to learn proper form or an experienced practitioner seeking to refine your technique, YogiSync provides personalized, real-time feedback anytime, anywhere.

### Inspiration

The inspiration for YogiSync came from our own challenges with maintaining proper form during yoga practice. We noticed that many practitioners, especially beginners, struggle to self-correct without immediate feedback from a teacher. Combining our passion for technology with a love for yoga, we envisioned a tool that could offer personalized guidance—making yoga more accessible and effective.


## ✨ Features

- 🎥 **Real-Time Pose Detection**: Uses MediaPipe to detect and track body landmarks from your webcam feed
- 🤖 **AI Pose Classification**: Custom-trained ML model (Random Forest) classifies various yoga poses with high accuracy
- 📊 **Instant Feedback**: Overlays skeleton visualizations on live video and provides corrective instructions for proper alignment
- 💬 **AI Yoga Coach Chatbot**: Powered by Google's Gemini LLM, recommends ideal yoga poses based on your needs and goals
- 📝 **Personalized Plans**: Generates customized yoga and dietary routines tailored to individual users
- 🎵 **Playlist Creation**: Creates yoga pose sequences optimized for your fitness level and objectives


## 🛠️ Tech Stack

### Backend
- **Python** - Core programming language
- **Flask** - Web framework for API endpoints and request handling
- **MediaPipe** - Real-time pose landmark detection
- **scikit-learn** - Machine learning model training (Random Forest)
- **Pickle** - Model serialization

### Frontend
- **HTML/CSS** - Structure and styling
- **JavaScript** - Interactive functionality
- **Bootstrap** - Responsive UI framework

### AI/ML
- **Google Gemini LLM** - Conversational AI yoga coach
- **Custom ML Model** - Pose classification trained on diverse dataset
- **MediaPipe Pose** - 33-point skeletal tracking


## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Webcam (for pose detection)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/yogisync.git
   cd yogisync
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Add your Gemini API key to .env
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser and navigate to**
   ```
   http://localhost:5000
   ```


## 🚀 Usage

1. **Start a Session**: Allow camera access when prompted
2. **Select a Pose**: Choose from the available yoga poses or ask the chatbot for recommendations
3. **Follow Instructions**: Position yourself in front of the camera and attempt the pose
4. **Receive Feedback**: Watch the skeleton overlay and read real-time corrective guidance
5. **Track Progress**: Monitor your form improvements over time
6. **Consult AI Coach**: Chat with the AI coach for personalized routines and dietary advice


## 🔬 How It Works

### 1. Data Collection
We collected a custom dataset of yoga poses featuring diversity in body types, lighting conditions, and camera angles to ensure robust model performance.

### 2. Pose Detection Pipeline
```
Webcam Feed → MediaPipe Pose → 33 Landmark Coordinates → Preprocessing
```

### 3. Pose Classification
```
Landmarks → Feature Extraction → Random Forest Model → Pose Classification
```

### 4. Feedback Generation
- Calculate joint angles for key body positions
- Compare against ideal pose parameters
- Generate specific corrective instructions
- Overlay visual feedback on video stream

### 5. AI Coach Integration
- Process user queries through Gemini LLM
- Generate contextual recommendations
- Create personalized workout plans
- Provide nutritional guidance



## 👥 Team

| Team Member | Role | Contribution |
|-------------|------|--------------|
| **Tanvi Chandan** | UI/UX Design & Frontend | Designed intuitive interfaces with strong focus on usability and led frontend integration |
| **Anusha Seshadri** | Full Stack Development | Developed LLM-based yoga coach and chatbot, contributed to frontend and backend integration |
| **Shubh Sehgal** | Backend Lead | Spearheaded backend development, integrated MediaPipe, trained ML models (Random Forest achieved lowest test loss), and developed APIs |
| **Iyashi Pal** | Backend Development | Integrated custom ML model for pose classification, implemented corrective feedback logic, and managed MediaPipe pipeline |




## 🙏 Acknowledgments

- Google MediaPipe team for the excellent pose detection library
- BrickHack 11 organizers and mentors
- The yoga community for inspiration and feedback
- Google Gemini for powering our AI coach


## 📞 Contact

Project Link: [https://devpost.com/software/yogisync](https://devpost.com/software/yogisync)



<div align="center">
  
**Made with ❤️ at BrickHack 11**

</div>
