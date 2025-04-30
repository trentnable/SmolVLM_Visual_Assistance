# test_smolvlm.py

# Import necessary libraries
import sys  # For accessing command-line arguments
import torch  # For handling tensor computations and checking GPU availability
from PIL import Image  # For loading and manipulating image files
from transformers import AutoProcessor, AutoModelForVision2Seq  # Hugging Face tools for vision-language models
import re  # For text processing (removing unwanted HTML tags)

# Step 1: Load the image file path from the command-line argument
# Example usage: python test_smolvlm.py sample.jpg
img_path = sys.argv[1]

# Load the image and ensure it is in RGB format (some images might be grayscale or CMYK)
image = Image.open(img_path).convert("RGB")

# Step 2: Set the device for running the model
# Use GPU ("cuda") if available, otherwise fall back to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Step 3: Load the pretrained processor and model
# The processor handles image and text preprocessing
# The model is SmolVLM-500M, a vision-to-language generation model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)  # Move model to the selected device (GPU or CPU)

# Step 4: Define the natural language prompt to guide the model's response
# This prompt is designed for assistive vision tasks and ensures focused, useful descriptions
prompt = (
    "<image> You are an assistive vision AI. Provide exactly one concise paragraph that: "
    "1) names the primary objects and their relative positions (left, center, right), "
    "2) gives quantities if relevant, "
    "3) calls out any obstacles or hazards, "
    "4) highlights distinctive colors or features. "
    "Do not use headings, labels, example lines, or numbers—just the plain description."
)

# Prepare the input for the model by processing the text prompt and image
# The processor converts them into tensor formats expected by the model
inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(DEVICE)

# Step 5: Generate the output from the model without computing gradients
# model.generate() is the function that produces text based on input tokens
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,  # Maximum number of tokens to generate in the output
        num_beams=1,         # Using greedy decoding (no beam search)
        do_sample=False      # Sampling disabled to make output deterministic
    )

# Step 6: Decode the tensor output into readable text using the processor
# batch_decode returns a list, so we access the first element
raw = processor.batch_decode(outputs, skip_special_tokens=True)[0]

# Step 7: Remove the prompt from the output if the model repeated it
# This is a common occurrence with instruction-following models
if raw.startswith(prompt):
    raw = raw[len(prompt):].strip()

# Step 8: Remove any HTML tags like <image>, <p>, etc., from the output
# This ensures the final caption is clean and ready to display
caption = re.sub(r'<[^>]+>', '', raw)

# Step 9: Print the final caption to the console
print(caption)
