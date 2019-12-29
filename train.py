# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/4/10 16:51


from __future__ import print_function, absolute_import, division

import numpy as np
import pandas as pd
from keras import layers, Model
from keras.callbacks import LearningRateScheduler
from keras.preprocessing import text, sequence

EMBEDDING_FILES = [
    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
    '../input/glove840b300dtxt/glove.840B.300d.txt'
]

NUM_MODELS = 2
BATCH_SIZE = 512
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
EPOCHS = 4
MAX_LEN = 220


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
    return embedding_matrix


def preprocess(data):
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
    return data


def bi_lstm_model(input_shape, embedding_matrix, lstm_units, num_aux_targets):
    inp = layers.Input(shape=input_shape)
    x = layers.Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(inp)
    x = layers.SpatialDropout1D(0.3)(x)
    x = layers.Bidirectional(layers.CuDNNLSTM(lstm_units, return_sequences=True))(x)
    x = layers.Bidirectional(layers.CuDNNLSTM(lstm_units, return_sequences=True))(x)

    hidden = layers.concatenate([layers.GlobalMaxPool1D()(x), layers.GlobalAveragePooling1D()(x)])
    hidden = layers.add([hidden, layers.Dense(4 * lstm_units, activation='relu')(hidden)])
    hidden = layers.add([hidden, layers.Dense(4 * lstm_units, activation='relu')(hidden)])

    out_1 = layers.Dense(1, activation='sigmoid')(hidden)
    out_2 = layers.Dense(num_aux_targets, activation='sigmoid')(hidden)

    model = Model(inputs=inp, outputs=[out_1, out_2])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

x_train = preprocess(train['comment_text'])
y_train = np.where(train['target'] >= 0.5, 1, 0)
y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
x_test = preprocess(test['comment_text'])

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(x_train) + list(x_test))
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

embedding_matrix = np.concatenate([build_matrix(tokenizer.index_word, f) for f in EMBEDDING_FILES], axis=-1)

checkpoint_predictions = []
weights = []

for model_idx in range(NUM_MODELS):
    model = bi_lstm_model(input_shape=(MAX_LEN,), embedding_matrix=embedding_matrix, lstm_units=LSTM_UNITS,
                          num_aux_targets=y_aux_train.shape[-1])
    for epoch_idx in range(EPOCHS):
        model.fit(x_train, [y_train, y_aux_train], batch_size=BATCH_SIZE, verbose=2,
                  callbacks=[LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** epoch_idx))])
        checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())
        weights.append(2 ** epoch_idx)

predictions = np.average(checkpoint_predictions, weights=weights, axis=0)
submission = pd.DataFrame.from_dict({
    'id': test['id'],
    'prediction': predictions
})
submission.to_csv('submission.csv', index=False)
