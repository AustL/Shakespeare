import numpy as np
import pickle
from keras.models import load_model

INDEXES_FILENAME = 'indexes.pickle'


def write(length):
    string = [indexes[c] for c in data[0]]
    start = np.array([indexes[c] for c in string]).reshape(1, 10, 1)
    prediction = model.predict(start)

    while len(string) < length:
        string.append(indexes[np.argmax(prediction)])
        x = np.array([indexes[c] for c in string[-10:]]).reshape(1, 10, 1)
        prediction = model.predict(x)

    return ' '.join(string)


data = np.load('data.npy')
with open(INDEXES_FILENAME, 'rb') as f:
    indexes = pickle.load(f)

model = load_model('models/train1/RNN_Final-53-0.021.model')

print(write(50))
