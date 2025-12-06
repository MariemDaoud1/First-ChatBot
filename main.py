import random
import json
import pickle  
import numpy as np
import os

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmtizer = WordNetLemmatizer()
base_dir = os.path.dirname(__file__)

words = pickle.load(open(os.path.join(base_dir, 'words.pkl'), 'rb'))
classes = pickle.load(open(os.path.join(base_dir, 'classes.pkl'), 'rb'))
model = load_model(os.path.join(base_dir, 'chatbot_model.h5'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmtizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

print("Chatbot is ready to talk!")
while True:
    message = input("You: ")
    ints = predict_class(message)
    if ints:
        print(f"Predicted intent: {ints[0]['intent']} with probability {ints[0]['probability']}")
    else:
        print("I didn't understand that. Can you try again?")