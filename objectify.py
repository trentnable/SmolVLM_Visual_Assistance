import re
import torch
import torch.nn.functional as F
from transformers import pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Object Labels
label_texts_object = {
    "0": "person,human,man,woman,child",
    "1": "bicycle,bike",
    "2": "car,automobile,vehicle",
    "3": "motorcycle,motorbike",
    "4": "airplane,plane,jet",
    "5": "bus",
    "6": "train,locomotive",
    "7": "truck,pickup,lorry",
    "8": "boat,ship,vessel",
    "9": "traffic light,street light",
    "10": "fire hydrant,hydrant",
    "11": "stop sign",
    "12": "parking meter",
    "13": "bench,park bench",
    "14": "bird",
    "15": "cat,feline",
    "16": "dog,canine",
    "17": "horse",
    "18": "sheep,lamb",
    "19": "cow,cattle",
    "20": "elephant",
    "21": "bear",
    "22": "zebra",
    "23": "giraffe",
    "24": "backpack",
    "25": "umbrella",
    "26": "handbag",
    "27": "tie",
    "28": "suitcase",
    "29": "frisbee",
    "30": "skis",
    "31": "snowboard",
    "32": "sports ball",
    "33": "kite",
    "34": "baseball bat",
    "35": "baseball glove",
    "36": "skateboard",
    "37": "surfboard",
    "38": "tennis racket",
    "39": "bottle",
    "40": "wine glass",
    "41": "cup",
    "42": "fork",
    "43": "knife",
    "44": "spoon",
    "45": "bowl",
    "46": "banana",
    "47": "apple",
    "48": "sandwich",
    "49": "orange",
    "50": "broccoli",
    "51": "carrot",
    "52": "hot dog",
    "53": "pizza",
    "54": "donut",
    "55": "cake",
    "56": "chair",
    "57": "couch",
    "58": "potted plant",
    "59": "bed",
    "60": "dining table",
    "61": "toilet",
    "62": "TV",
    "63": "laptop",
    "64": "mouse",
    "65": "remote",
    "66": "keyboard",
    "67": "cell phone,mobile phone,phone",
    "68": "microwave",
    "69": "oven",
    "70": "toaster",
    "71": "sink",
    "72": "refrigerator,fridge",
    "73": "book",
    "74": "clock",
    "75": "vase",
    "76": "scissors",
    "77": "teddy bear",
    "78": "hair drier",
    "79": "toothbrush",
    "unlisted_object": "Request includes an object not listed"
}

def normalize(text):
    """Lowercase,remove punctuation"""
    return re.sub(r'[^a-zA-Z0-9 ]+','',text.lower()).strip()

def exact_keyword_match(user_text, select):
    """Return all label keywords that appear in the speech output"""
    text = normalize(user_text)
    matches = []

    if select == "mode":
        for k, v in label_texts_mode.items():
            # Clean up and split keywords
            keywords = [kw.strip() for kw in v.lower().split(",")]
            for kw in keywords:
                kw = normalize(kw)
                if kw in text:
                    match = k
                    return match

        return matches if matches else None
    
    elif select == "object":
        for k, v in label_texts_object.items():
            # Clean up and split keywords
            keywords = [kw.strip() for kw in v.lower().split(",")]
            for kw in keywords:
                kw = normalize(kw)
                if kw in text and k not in matches:
                    matches.append(k)

        return matches if matches else None

def classify_request(user_text):
    """ Keyword match """
    # exact keyword match
    match = exact_keyword_match(user_text, "object")
    if match:
        return match
    else:
        return "unlisted_object"
    
#Mode Labels
label_texts_mode = {
    "one": "find,locate,where,where's",
    "two": "read,reading,interpret,say,words",
}


def mode_select(user_text):
    """ Classify a user command into one of the predefined modes """
    
    # Handle empty input
    if not user_text or user_text.strip() == "":
        print(f"(empty input)")
        return "null"
    
    # exact keyword match
    match = exact_keyword_match(user_text, "mode")
    if match:
        return match
    else:
        return "null"