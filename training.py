import random
import json
import os
import pickle
import numpy as np

import nltk
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmtizer = WordNetLemmatizer()

base_dir = os.path.dirname(__file__)            

path = os.path.join(base_dir, 'intents.json')

try:
    with open(path, 'r', encoding='utf-8') as f:
        intents = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Fichier introuvable: {path}")
except json.JSONDecodeError as e:
    raise ValueError(f"JSON invalide dans {path}: {e}")

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmtizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open(os.path.join(base_dir, 'words.pkl'), 'wb'))
pickle.dump(classes, open(os.path.join(base_dir, 'classes.pkl'), 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmtizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save(os.path.join(base_dir, 'chatbot_model.h5'))
print("Modèle entraîné et sauvegardé sous 'chatbot_model.h5'")