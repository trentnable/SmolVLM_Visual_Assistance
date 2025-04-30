# test_smolvlm.py
import sys
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import re

# 1. load image from first CLI arg
img_path = sys.argv[1]
image = Image.open(img_path).convert("RGB")

# 2. set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 3. load processor & model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32
).to(DEVICE)  # :contentReference[oaicite:0]{index=0}

# 4. build your prompt + inputs
prompt = (
    "<image> You are an assistive vision AI. Provide exactly one concise paragraph that: "
    "1) names the primary objects and their relative positions (left, center, right), "
    "2) gives quantities if relevant, "
    "3) calls out any obstacles or hazards, "
    "4) highlights distinctive colors or features. "
    "Do not use headings, labels, example lines, or numbers—just the plain description."
)

inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(DEVICE)

# 5. generate and print
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200, 
        num_beams=1,
        do_sample=False
    )

raw = processor.batch_decode(outputs, skip_special_tokens=True)[0]

# strip any accidental echo of the prompt
if raw.startswith(prompt):
    raw = raw[len(prompt):].strip()

# remove any HTML tags
caption = re.sub(r'<[^>]+>', '', raw)

print(caption)


