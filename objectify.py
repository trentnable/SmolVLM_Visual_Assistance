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
    "0": "COCO object class: person,human,man,woman,child",
    "1": "COCO object class: bicycle,bike",
    "2": "COCO object class: car,automobile,vehicle",
    "3": "COCO object class: motorcycle,motorbike",
    "4": "COCO object class: airplane,plane,jet",
    "5": "COCO object class: bus",
    "6": "COCO object class: train,locomotive",
    "7": "COCO object class: truck,pickup,lorry",
    "8": "COCO object class: boat,ship,vessel",
    "9": "COCO object class: traffic light,street light",
    "10": "COCO object class: fire hydrant,hydrant",
    "11": "COCO object class: stop sign",
    "12": "COCO object class: parking meter",
    "13": "COCO object class: bench,park bench",
    "14": "COCO object class: bird",
    "15": "COCO object class: cat,feline",
    "16": "COCO object class: dog,canine",
    "17": "COCO object class: horse",
    "18": "COCO object class: sheep,lamb",
    "19": "COCO object class: cow,cattle",
    "20": "COCO object class: elephant",
    "21": "COCO object class: bear",
    "22": "COCO object class: zebra",
    "23": "COCO object class: giraffe",
    "24": "COCO object class: backpack",
    "25": "COCO object class: umbrella",
    "26": "COCO object class: handbag",
    "27": "COCO object class: tie",
    "28": "COCO object class: suitcase",
    "29": "COCO object class: frisbee",
    "30": "COCO object class: skis",
    "31": "COCO object class: snowboard",
    "32": "COCO object class: sports ball",
    "33": "COCO object class: kite",
    "34": "COCO object class: baseball bat",
    "35": "COCO object class: baseball glove",
    "36": "COCO object class: skateboard",
    "37": "COCO object class: surfboard",
    "38": "COCO object class: tennis racket",
    "39": "COCO object class: bottle",
    "40": "COCO object class: wine glass",
    "41": "COCO object class: cup",
    "42": "COCO object class: fork",
    "43": "COCO object class: knife",
    "44": "COCO object class: spoon",
    "45": "COCO object class: bowl",
    "46": "COCO object class: banana",
    "47": "COCO object class: apple",
    "48": "COCO object class: sandwich",
    "49": "COCO object class: orange",
    "50": "COCO object class: broccoli",
    "51": "COCO object class: carrot",
    "52": "COCO object class: hot dog",
    "53": "COCO object class: pizza",
    "54": "COCO object class: donut",
    "55": "COCO object class: cake",
    "56": "COCO object class: chair",
    "57": "COCO object class: couch",
    "58": "COCO object class: potted plant",
    "59": "COCO object class: bed",
    "60": "COCO object class: dining table",
    "61": "COCO object class: toilet",
    "62": "COCO object class: TV",
    "63": "COCO object class: laptop",
    "64": "COCO object class: mouse",
    "65": "COCO object class: remote",
    "66": "COCO object class: keyboard",
    "67": "COCO object class: cell phone",
    "68": "COCO object class: microwave",
    "69": "COCO object class: oven",
    "70": "COCO object class: toaster",
    "71": "COCO object class: sink",
    "72": "COCO object class: refrigerator",
    "73": "COCO object class: book",
    "74": "COCO object class: clock",
    "75": "COCO object class: vase",
    "76": "COCO object class: scissors",
    "77": "COCO object class: teddy bear",
    "78": "COCO object class: hair drier",
    "79": "COCO object class: toothbrush"
}



label_embeddings = {k: model.encode(v,convert_to_tensor=True,normalize_embeddings=True,device=DEVICE)
                    for k,v in label_texts.items()}


def normalize(text):
    """Lowercase,remove punctuation"""
    return re.sub(r'[^a-zA-Z0-9 ]+','',text.lower()).strip()

def exact_keyword_match(user_text):
    text = normalize(user_text)
    for k,v in label_texts.items():
        keywords = [kw.strip() for kw in v.lower().replace("coco object class:","").split(",")]
        for kw in keywords:
            kw = normalize(kw)
            if kw in text:
                return k
    return None


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
    "one": "find or locate a physical object or person from COCO classes",
    "two": "read or interpret written text,signs,or labels"
}

label_embeddings_mode = {k: model.encode(v,convert_to_tensor=True,normalize_embeddings=True,device=DEVICE)
                         for k,v in label_texts_mode.items()}

def mode_select(user_text):
    emb = model.encode(user_text,convert_to_tensor=True,normalize_embeddings=True,device=DEVICE)
    scores = {k: util.cos_sim(emb,v).item() for k,v in label_embeddings_mode.items()}
    return max(scores,key=scores.get)