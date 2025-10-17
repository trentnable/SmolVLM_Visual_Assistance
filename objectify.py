import re
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# mixedbread model
EMBED_MODEL_ID = "mixedbread-ai/mxbai-embed-large-v1"
model = SentenceTransformer(EMBED_MODEL_ID).to(DEVICE)


# Labels
label_texts = {
    "0": "COCO class: person,human,man,woman,child",
    "1": "COCO class: bicycle,bike",
    "2": "COCO class: car,automobile,vehicle",
    "3": "COCO class: motorcycle,motorbike",
    "4": "COCO class: airplane,plane,jet",
    "5": "COCO class: bus",
    "6": "COCO class: train,locomotive",
    "7": "COCO class: truck,pickup,lorry",
    "8": "COCO class: boat,ship,vessel",
    "9": "COCO class: traffic light,street light",
    "10": "COCO class: fire hydrant,hydrant",
    "11": "COCO class: stop sign",
    "12": "COCO class: parking meter",
    "13": "COCO class: bench,park bench",
    "14": "COCO class: bird",
    "15": "COCO class: cat,feline",
    "16": "COCO class: dog,canine",
    "17": "COCO class: horse",
    "18": "COCO class: sheep,lamb",
    "19": "COCO class: cow,cattle",
    "20": "COCO class: elephant",
    "21": "COCO class: bear",
    "22": "COCO class: zebra",
    "23": "COCO class: giraffe",
    "24": "COCO class: backpack",
    "25": "COCO class: umbrella",
    "26": "COCO class: handbag",
    "27": "COCO class: tie",
    "28": "COCO class: suitcase",
    "29": "COCO class: frisbee",
    "30": "COCO class: skis",
    "31": "COCO class: snowboard",
    "32": "COCO class: sports ball",
    "33": "COCO class: kite",
    "34": "COCO class: baseball bat",
    "35": "COCO class: baseball glove",
    "36": "COCO class: skateboard",
    "37": "COCO class: surfboard",
    "38": "COCO class: tennis racket",
    "39": "COCO class: bottle",
    "40": "COCO class: wine glass",
    "41": "COCO class: cup",
    "42": "COCO class: fork",
    "43": "COCO class: knife",
    "44": "COCO class: spoon",
    "45": "COCO class: bowl",
    "46": "COCO class: banana",
    "47": "COCO class: apple",
    "48": "COCO class: sandwich",
    "49": "COCO class: orange",
    "50": "COCO class: broccoli",
    "51": "COCO class: carrot",
    "52": "COCO class: hot dog",
    "53": "COCO class: pizza",
    "54": "COCO class: donut",
    "55": "COCO class: cake",
    "56": "COCO class: chair",
    "57": "COCO class: couch",
    "58": "COCO class: potted plant",
    "59": "COCO class: bed",
    "60": "COCO class: dining table",
    "61": "COCO class: toilet",
    "62": "COCO class: TV",
    "63": "COCO class: laptop",
    "64": "COCO class: mouse",
    "65": "COCO class: remote",
    "66": "COCO class: keyboard",
    "67": "COCO class: cell phone",
    "68": "COCO class: microwave",
    "69": "COCO class: oven",
    "70": "COCO class: toaster",
    "71": "COCO class: sink",
    "72": "COCO class: refrigerator",
    "73": "COCO class: book",
    "74": "COCO class: clock",
    "75": "COCO class: vase",
    "76": "COCO class: scissors",
    "77": "COCO class: teddy bear",
    "78": "COCO class: hair drier",
    "79": "COCO class: toothbrush"
}



label_embeddings = {k: model.encode(v,convert_to_tensor=True,normalize_embeddings=True,device=DEVICE)
                    for k,v in label_texts.items()}


def normalize(text):
    """Lowercase,remove punctuation"""
    return re.sub(r'[^a-zA-Z0-9 ]+','',text.lower()).strip()

def exact_keyword_match(user_text):
    """Return all label keywords that appear in the speech output"""
    text = normalize(user_text)
    matches = []

    for k, v in label_texts.items():
        # Clean up and split keywords
        keywords = [kw.strip() for kw in v.lower().replace("coco class:", "").split(",")]
        for kw in keywords:
            kw = normalize(kw)
            if kw in text and k not in matches:
                matches.append(k)

    return matches if matches else None



def classify_request(user_text):
    # exact keyword match
    match = exact_keyword_match(user_text)
    if match:
        return match

    # fallback to embedding similarity
    emb = model.encode(user_text,convert_to_tensor=True,normalize_embeddings=True,device=DEVICE)
    scores = {k: util.cos_sim(emb,v).item() for k,v in label_embeddings.items()}
    return max(scores,key=scores.get)


label_texts_mode = {
    "one": "find or locate a physical or person from COCO classes",
    "two": "read or interpret written text,signs,or labels"
}

label_embeddings_mode = {k: model.encode(v,convert_to_tensor=True,normalize_embeddings=True,device=DEVICE)
                         for k,v in label_texts_mode.items()}

def mode_select(user_text):
    emb = model.encode(user_text,convert_to_tensor=True,normalize_embeddings=True,device=DEVICE)
    scores = {k: util.cos_sim(emb,v).item() for k,v in label_embeddings_mode.items()}
    return max(scores,key=scores.get)