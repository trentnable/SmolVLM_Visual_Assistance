import time



def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
        # Get last hidden state: shape [batch, seq_len, hidden_dim]
        hidden_states = outputs.hidden_states[-1]
        # Average pooling over the sequence dimension
        embedding = hidden_states.mean(dim=1).squeeze().cpu()
    return embedding

label_texts = {
    "1": "The request is about a person, someone nearby, or people in general. Example requests: 'Where is the person?', 'Find the man/woman/child', 'Show me where people are.'",
    "2": "The request is about a bicycle, also called a bike or cycle. Example requests: 'Where is my bike?', 'Find the bicycle.'",
    "3": "The request is about a car or automobile. Example requests: 'Where is my car?', 'Show me the car in the picture.'",
    "4": "The request is about a motorcycle or motorbike. Example requests: 'Where is the motorcycle?', 'Find the motorbike.'",
    "5": "The request is about an airplane, plane, or jet. Example requests: 'Where is the airplane?', 'Find the plane in the sky.'",
    "6": "The request is about a bus, such as a city bus or school bus. Example requests: 'Where is the bus?', 'Find the school bus.'",
    "7": "The request is about a train or locomotive. Example requests: 'Where is the train?', 'Show me the train.'",
    "8": "The request is about a truck, lorry, or pickup. Example requests: 'Where is the truck?', 'Find the pickup truck.'",
    "9": "The request is about a boat, ship, or watercraft. Example requests: 'Where is the boat?', 'Find the ship on the water.'",
    "10": "The request is about a traffic light or stoplight. Example requests: 'Where is the traffic light?', 'Show me the stoplight.'",
    "11": "The request is about a fire hydrant, often seen on streets. Example requests: 'Where is the hydrant?', 'Find the fire hydrant.'",
    "13": "The request is about a stop sign. Example requests: 'Where is the stop sign?', 'Show me the stop sign on the road.'",
    "14": "The request is about a parking meter. Example requests: 'Where is the parking meter?', 'Find the meter for parking.'",
    "15": "The request is about a bench, such as a park bench. Example requests: 'Where is the bench?', 'Find a place to sit.'",
    "16": "The request is about a bird. Example requests: 'Where is the bird?', 'Show me the bird flying or sitting.'",
    "17": "The request is about a cat or kitten. Example requests: 'Where is the cat?', 'Find my kitty.'",
    "18": "The request is about a dog or puppy. Example requests: 'Where is the dog?', 'Find my puppy.'",
    "19": "The request is about a horse. Example requests: 'Where is the horse?', 'Find the horse in the field.'",
    "20": "The request is about a sheep or lamb. Example requests: 'Where is the sheep?', 'Find the lamb.'",
    "21": "The request is about a cow. Example requests: 'Where is the cow?', 'Find the cow in the field.'",
    "22": "The request is about an elephant. Example requests: 'Where is the elephant?', 'Find the elephant in the zoo.'",
    "23": "The request is about a bear. Example requests: 'Where is the bear?', 'Find the bear in the picture.'",
    "24": "The request is about a zebra. Example requests: 'Where is the zebra?', 'Find the striped animal.'",
    "25": "The request is about a giraffe. Example requests: 'Where is the giraffe?', 'Find the tall animal with a long neck.'",
    "27": "The request is about a backpack, rucksack, or school bag. Example requests: 'Where is my backpack?', 'Find the bag I carry on my back.'",
    "28": "The request is about an umbrella. Example requests: 'Where is the umbrella?', 'Find the thing for rain.'",
    "31": "The request is about a handbag, purse, or bag. Example requests: 'Where is my purse?', 'Find the handbag.'",
    "32": "The request is about a tie or necktie. Example requests: 'Where is the tie?', 'Find the necktie.'",
    "33": "The request is about a suitcase or luggage. Example requests: 'Where is my suitcase?', 'Find the luggage bag.'",
    "34": "The request is about a frisbee or flying disc. Example requests: 'Where is the frisbee?', 'Find the flying disc.'",
    "35": "The request is about skis. Example requests: 'Where are the skis?', 'Find the ski equipment.'",
    "36": "The request is about a snowboard. Example requests: 'Where is the snowboard?', 'Find the snow board for winter sports.'",
    "37": "The request is about a sports ball, like soccer ball, basketball, or football. Example requests: 'Where is the ball?', 'Find the sports ball.'",
    "38": "The request is about a kite. Example requests: 'Where is the kite?', 'Find the kite flying.'",
    "39": "The request is about a baseball bat. Example requests: 'Where is the bat?', 'Find the baseball bat.'",
    "40": "The request is about a baseball glove or mitt. Example requests: 'Where is the glove?', 'Find the baseball mitt.'",
    "41": "The request is about a skateboard. Example requests: 'Where is the skateboard?', 'Find the board for skating.'",
    "42": "The request is about a surfboard. Example requests: 'Where is the surfboard?', 'Find the board for surfing.'",
    "43": "The request is about a tennis racket. Example requests: 'Where is the tennis racket?', 'Find the racket.'",
    "44": "The request is about a bottle, such as a water bottle. Example requests: 'Where is my bottle?', 'Find the drink bottle.'",
    "46": "The request is about a wine glass. Example requests: 'Where is the wine glass?', 'Find the glass for wine.'",
    "47": "The request is about a cup or mug. Example requests: 'Where is my cup?', 'Find the coffee mug.'",
    "48": "The request is about a fork. Example requests: 'Where is the fork?', 'Find the eating utensil.'",
    "49": "The request is about a knife. Example requests: 'Where is the knife?', 'Find the kitchen knife.'",
    "50": "The request is about a spoon. Example requests: 'Where is the spoon?', 'Find the spoon for eating.'",
    "51": "The request is about a bowl. Example requests: 'Where is the bowl?', 'Find the soup bowl.'",
    "52": "The request is about a banana. Example requests: 'Where is the banana?', 'Find the fruit.'",
    "53": "The request is about an apple. Example requests: 'Where is the apple?', 'Find the fruit.'",
    "54": "The request is about a sandwich. Example requests: 'Where is the sandwich?', 'Find my food.'",
    "55": "The request is about an orange. Example requests: 'Where is the orange?', 'Find the orange fruit.'",
    "56": "The request is about broccoli. Example requests: 'Where is the broccoli?', 'Find the green vegetable.'",
    "57": "The request is about a carrot. Example requests: 'Where is the carrot?', 'Find the orange vegetable.'",
    "58": "The request is about a hot dog. Example requests: 'Where is the hot dog?', 'Find the sausage in a bun.'",
    "59": "The request is about pizza. Example requests: 'Where is the pizza?', 'Find the slice of pizza.'",
    "60": "The request is about a donut. Example requests: 'Where is the donut?', 'Find the doughnut snack.'",
    "61": "The request is about a cake. Example requests: 'Where is the cake?', 'Find the birthday cake.'",
    "62": "The request is about a chair. Example requests: 'Where is the chair?', 'Find a place to sit.'",
    "63": "The request is about a couch or sofa. Example requests: 'Where is the couch?', 'Find the sofa.'",
    "64": "The request is about a potted plant or houseplant. Example requests: 'Where is the plant?', 'Find the potted plant.'",
    "65": "The request is about a bed. Example requests: 'Where is the bed?', 'Find the bed in the room.'",
    "67": "The request is about a dining table. Example requests: 'Where is the dining table?', 'Find the table for eating.'",
    "70": "The request is about a toilet. Example requests: 'Where is the toilet?', 'Find the bathroom toilet.'",
    "72": "The request is about a TV, television, or screen. Example requests: 'Where is the TV?', 'Find the television.'",
    "73": "The request is about a laptop computer. Example requests: 'Where is my laptop?', 'Find the computer.'",
    "74": "The request is about a computer mouse (not the animal). Example requests: 'Where is my mouse?', 'Find the computer mouse.'",
    "75": "The request is about a remote control. Example requests: 'Where is the remote?', 'Find the TV controller.'",
    "76": "The request is about a keyboard for a computer. Example requests: 'Where is the keyboard?', 'Find the typing keyboard.'",
    "77": "The request is about a cell phone, smartphone, or mobile phone. Example requests: 'Where is my phone?', 'Find the smartphone.'",
    "78": "The request is about a microwave oven. Example requests: 'Where is the microwave?', 'Find the microwave oven.'",
    "79": "The request is about an oven or stove oven. Example requests: 'Where is the oven?', 'Find the kitchen oven.'",
    "80": "The request is about a toaster. Example requests: 'Where is the toaster?', 'Find the bread toaster.'",
    "81": "The request is about a sink, such as a kitchen sink. Example requests: 'Where is the sink?', 'Find the bathroom sink.'",
    "82": "The request is about a refrigerator or fridge. Example requests: 'Where is the fridge?', 'Find the refrigerator.'",
    "84": "The request is about a book. Example requests: 'Where is the book?', 'Find my reading book.'",
    "85": "The request is about a clock. Example requests: 'Where is the clock?', 'Find the wall clock.'",
    "86": "The request is about a vase. Example requests: 'Where is the vase?', 'Find the flower vase.'",
    "87": "The request is about scissors. Example requests: 'Where are the scissors?', 'Find the cutting scissors.'",
    "88": "The request is about a teddy bear or stuffed toy. Example requests: 'Where is the teddy bear?', 'Find the stuffed animal.'",
    "89": "The request is about a hair drier or blow dryer. Example requests: 'Where is the hair dryer?', 'Find the blow dryer.'",
    "90": "The request is about a toothbrush. Example requests: 'Where is the toothbrush?', 'Find the brush for teeth.'"
}

label_embeddings = {k: get_embedding(v) for k, v in label_texts.items()}

import torch.nn.functional as F

def classify_request(user_text):

      emb = get_embedding(user_text)
      scores = {k: F.cosine_similarity(emb, v, dim=0).item() 
              for k, v in label_embeddings.items()}
      return max(scores, key=scores.get)
    
# start_time = time.time()

req1 = "Where's my bottle of water?"
print(req1)

label = classify_request(req1)

print(label)

# end_time = time.time()

# Calculate elapsed time
# elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time:.4f} seconds")