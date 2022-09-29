import os
from collections import deque
import numpy as np
import pickle
from keras.layers import LSTM, Dense, BatchNormalization, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

TEXT_FILENAME = 'plays.txt'
DATA_FILENAME = 'data.npy'
INDEXES_FILENAME = 'indexes.pickle'
LOG_DIR = 'logs'

TIME_STEPS = 10
MIN_FREQUENCY = 10
TEST_SPLIT = 0.1


def processData():
    if DATA_FILENAME not in os.listdir():
        train = deque()
        indexes = {}
        frequency = {}

        with open(TEXT_FILENAME, 'r') as f:
            for line in f:
                line = line.lower().split()

                for word in line:
                    word = ''.join(c for c in word if c.isalnum())

                    if word:
                        train.append(word)
                        frequency[word] = frequency.get(word, 0) + 1

        ignored = {word for word in frequency.keys() if frequency[word] < MIN_FREQUENCY}
        train = [word for word in train if word not in ignored]

        indexes.update([(word, i) for i, word in enumerate(set(train))])
        indexes.update([(i, word) for i, word in enumerate(set(train))])
        train = np.array([indexes[word] for word in train])
        train = train[:len(train) - len(train) % TIME_STEPS]
        train = np.array(np.split(train, len(train) // TIME_STEPS))

        np.save(DATA_FILENAME, train)

        with open(INDEXES_FILENAME, 'wb') as f:
            pickle.dump(indexes, f, protocol=pickle.HIGHEST_PROTOCOL)


processData()

data = np.load(DATA_FILENAME)
with open(INDEXES_FILENAME, 'rb') as f:
    indexes = pickle.load(f)

splitIndex = int(len(data) * TEST_SPLIT)
vocabSize = len(indexes) // 2

train = data[splitIndex:]
train = train[:(len(train) - len(train) % TIME_STEPS) + 1]
trainX = train[:-1].reshape(len(train) - 1, TIME_STEPS, 1)
trainY = np.array([sentence[0] for sentence in train[1:]])

test = data[:splitIndex]
test = test[:(len(test) - len(test) % (1 * TIME_STEPS)) + 1]
testX = test[:-1].reshape(len(test) - 1, TIME_STEPS, 1)
testY = np.array([sentence[0] for sentence in test[1:]])

filepath = 'RNN_Shuffle-{epoch:02d}'
checkpoint = ModelCheckpoint('models/train2/{}.model'.format(
    filepath, monitor='val_loss', verbose=1, mode='min')
)
best = ModelCheckpoint('best.model'.format(
    filepath, monitor='val_loss', verbose=1, mode='min', save_best_only=True)
)
stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)

model = Sequential()
model.add(LSTM(128, input_shape=trainX.shape[1:], return_sequences=True, activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True, activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(LSTM(128, activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(vocabSize, activation='softmax'))

model.summary()

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=trainX, y=trainY, batch_size=1024, epochs=50, validation_data=(testX, testY),
          callbacks=[checkpoint, best, stop], shuffle=True)

model.save('model.h5')


def output(i):
    x = testX[i].reshape(1, 10, 1)
    prediction = model.predict(x)

    print('Sentence:', ' '.join(''.join(indexes[i] for i in sentence) for sentence in x[0]))
    print('Predicted:', indexes[np.argmax(prediction)])
    print('Expected:', indexes[testX[i + 1][0][0]])


output(0)
