# Visual_Assistance Project


This project aims to provide visual assistance using computer vision and speech technologies. It leverages object detection (YOLO), potentially depth estimation (MiDaS), speech recognition (Whisper), and text-to-speech (gTTS) to interact with the user and describe the visual environment or perform specific vision-related tasks based on voice commands.

---

## 📋 Prerequisites

Before setting up the project, you'll need a few essential tools installed on your Windows system.

### 1. **Python**
This project is written in Python. We recommend **Python 3.9, 3.10, or 3.11**.  
- **Why?** It's the programming language the code is written in.  
- **Download:** [python.org/downloads/](https://www.python.org/downloads/)  
- **Important:** During installation, make sure to check the box that says **"Add Python to PATH"**. This allows you to run Python from the command line easily.  

---

### 2. **Git**
This tool is needed to download (clone) the project code from GitHub.  
- **Why?** To get a copy of the project files onto your computer.  
- **Download:** [git-scm.com/downloads](https://git-scm.com/downloads)  
- You can generally accept the default settings during installation.  

---

### 3. **FFmpeg**
This is a multimedia framework required by the `openai-whisper` library for processing audio during speech recognition.  
- **Why?** Whisper needs it to decode audio input from your microphone.  
- **Download:** Get the **"essentials"** build zip from [gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/).  
- **Setup:**  
  - Extract the downloaded zip file to a permanent location (e.g., `C:\ffmpeg`).  
  - Add the `bin` folder from inside the extracted folder (e.g., `C:\ffmpeg\bin`) to your Windows **System PATH** environment variable.  
  - (Search Windows for "Edit the system environment variables" to access this setting).  

---

### 4. **(Potentially Needed) C++ Build Tools**
Some Python libraries, notably `PyAudio`, sometimes need C++ tools to compile correctly during installation on Windows if pre-built versions aren't available for your specific Python version.  
It's good to install these proactively if you encounter related errors.  

- **Why?** To compile certain Python packages from source code if direct installation fails.  
- **Download:** Get the **Build Tools for Visual Studio 2022** from [visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022).  
- **Setup:** Run the installer and select the **"Desktop development with C++"** workload. After installation, **restart your computer**.  

---

## ⚙️ Setup Instructions

Follow these steps using **Command Prompt (cmd)** or **PowerShell**.

---

### 1. Clone the Repository

First, navigate to where you want to store the project (e.g., your Documents folder) and download the code using Git.

```cmd
:: Navigate to your Documents folder (or preferred location)
cd %USERPROFILE%\Documents

:: Clone the project from GitHub
git clone https://github.com/trentnable/Visual_Assistance.git

:: Enter the newly created project directory
cd Visual_Assistance
```

---

### 2. Create and Activate a Virtual Environment

A virtual environment is crucial for keeping project dependencies separate from your global Python installation, preventing conflicts.

```cmd
:: Create a virtual environment named 'venv'
python -m venv venv

:: Activate the virtual environment
.\venv\Scripts\activate
```

After activation, your command prompt line should start with `(venv)`.  
All subsequent pip install commands must be run while this environment is active.

---

### 3. Install Dependencies (Step-by-Step)

We'll install the required Python libraries in logical groups.

---

#### a) Core Computer Vision Libraries
These are essential for image processing and object detection.  
`ultralytics` includes torch (PyTorch) as a dependency.

```cmd
:: Upgrade pip (good practice)
pip install --upgrade pip

:: Install OpenCV (for camera access and image manipulation) and Ultralytics (for YOLO object detection)
pip install opencv-python ultralytics
```

---

#### b) Speech Input Libraries
These libraries handle listening to your microphone and converting your speech to text.

```cmd
:: Install Whisper (speech-to-text), SoundDevice and PyAudio (microphone access)
pip install openai-whisper sounddevice PyAudio SpeechRecognition
```

> **Note on PyAudio:**  
> If `pip install PyAudio` fails with a C++ error, ensure you have installed the C++ Build Tools (Prerequisite #4) and restarted your PC. Then try installing PyAudio again.

---

#### c) Speech Output Libraries
These libraries convert the application's text responses into audible speech.

```cmd
:: Install Google Text-to-Speech (gTTS) and Pygame (for playing the audio)
pip install gTTS pygame
```

---

#### d) Helper & Model-Specific Libraries
These provide additional functionalities needed by the main scripts or specific models.

```cmd
:: Install Keyboard (for detecting key presses like 'm'), Sentence Transformers (for text comparison/understanding in objectify.py),
:: TIMM (used by MiDaS depth model), and PyPIWin32 (needed by pyttsx3 on Windows for older TTS engines, though gTTS is primary here)
pip install keyboard sentence-transformers timm pypiwin32 pyttsx3
```

> **Note:**  
> We install `pyttsx3` here as a fallback/alternative TTS engine potentially used in older parts of the code, even though `gTTS` seems primary.  
> `pypiwin32` supports `pyttsx3` on Windows.

---

## ▶️ Running the Application

Ensure your virtual environment is active (`(venv)` is visible). Then, run the main script:

```cmd
python headless_main.py
```

---

