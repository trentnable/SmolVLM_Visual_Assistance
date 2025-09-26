import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {DEVICE}...")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE)

def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
        # Get last hidden state: shape [batch, seq_len, hidden_dim]
        hidden_states = outputs.hidden_states[-1]
        # Average pooling over the sequence dimension
        embedding = hidden_states.mean(dim=1).squeeze().cpu()
    return embedding

label_texts1 = {
    "one": "The request is about reading.",
    "two": "The request is about finding the location of an object or place.",
}
label_embeddings1 = {k: get_embedding(v) for k, v in label_texts1.items()}

label_texts2 = {
    "close": "User says yes or gives affirming response",
    "far": "User is unsure, says no, or gives a denying response",
}
label_embeddings2 = {k: get_embedding(v) for k, v in label_texts2.items()}

import torch.nn.functional as F

def classify_request(user_text, select):
    if select == 1:
        emb = get_embedding(user_text)
        scores = {k: F.cosine_similarity(emb, v, dim=0).item() 
                for k, v in label_embeddings1.items()}
        return max(scores, key=scores.get)
    elif select == 2:
        emb = get_embedding(user_text)
        scores = {k: F.cosine_similarity(emb, v, dim=0).item() 
                for k, v in label_embeddings2.items()}
        return max(scores, key=scores.get)
    else:
        print("Request Selection classify_request Function Error")

uinput = "Where's my bottle of water?"

start_time = time.time()

req1 = "I need help. "+ uinput
print(req1)

select = 1
mode = classify_request(req1,select)

if mode == "one":
    print("Helping with reading")

elif mode == "two":
    print("Is it nearby?")
    req2 = "NO"
    print(req2)
    select = 2
    dist = classify_request(req2,select)

    if dist == "close":
        print("Helping with close task")

    elif dist == "far":
        print("Helping with direction and planning")

    else:
        print("mode2 selection error")

else:
    print("general mode selection error")

end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")
