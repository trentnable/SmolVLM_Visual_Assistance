#  SmolVLM Visual Assistance

##  Windows Setup Instructions

Clone the Repository
```bash
git clone -b landon https://github.com/trentnable/SmolVLM_Visual_Assistance.git
```

Move to new directory
```bash
cd SmolVLM_Visual_Assistance
```

Create python or conda virtual environment

```bash

```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv smartglasses-env
source smartglasses-env/bin/activate
```

>  If `venv` does not work, please make sure Python 3 is installed and you're not inside a conda environment.

### 3. Install the Required Packages

```bash
pip install -r requirements.txt
```

>  If `requirements.txt` is missing, you can install dependencies manually:

```bash
pip install torch torchvision transformers pillow
```

---

##  Running the Model

### 1. Add an image

Place any `.jpg` or `.png` image in the project directory, or use one of the provided samples (e.g., `sample.jpg`, `sample2.jpg`, etc.).

### 2. Run the model

```bash
python smol_test.py sample2.jpg
```

The script will generate a detailed, plain-text description of the image.

---

##  Prompt Design

This model uses a universal vision-language prompt:

```python
prompt = (
    "<image> You are an assistive vision AI. Provide exactly one concise paragraph that: "
    "1) names the primary objects and their relative positions (left, center, right), "
    "2) gives quantities if relevant, "
    "3) calls out any obstacles or hazards, "
    "4) highlights distinctive colors or features. "
    "Do not use headings, labels, example lines, or numbers—just the plain description."
)
```

The output is post-processed to strip out:
- Repeated prompts
- HTML tags
- Unfinished sentences

---

##  Output Example

```
People are sitting around a table. There are four people in the image. There are two women and two men. The women are smiling. The men are smiling. There are plates of food on the table. There are glasses of wine and juice on the table. There is a bowl of pasta on the table. There is a vase of flowers on the table. There is a chair behind the table. There is a window behind the table. There is a wall behind the table. There is a shelf above the table. There is a bowl on the shelf. There is a potted plant on the shelf. There is a clock on the wall. There is a picture on the wall. There is a mirror on the wall. There is a door on the wall. There is a rug on the floor. There is a chair in the corner of the room. There is a person standing in the corner of the room. There is a person sitting in the corner of the room.
```

---

##  Project Structure

```
SmolVLM_Visual_Assistance/
├── smol_test.py         # Main inference script
├── sample.jpg           # Example test image
├── sample2.jpg          # Example test image
├── sample3.jpg          # Example test image
├── requirements.txt     # Python dependencies
├── .gitignore           # Ignores venv, __pycache__, etc.
└── README.md            # Instructions
```

---

##  Credits

- **Model**: [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) by HuggingFaceTB  
- **Vision-Language Framework**: [Transformers](https://github.com/huggingface/transformers) by Hugging Face  
- **Script & Integration**: Senior Design Team

---

##  Sources provided by Dr. Anagnostopoulos:  
[Sources](SOURCES.md)