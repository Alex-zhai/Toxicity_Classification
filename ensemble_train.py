# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/4/11 15:26


from __future__ import print_function, absolute_import, division

import numpy as np
import pandas as pd
from tensorflow.keras import layers, Model
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing import text, sequence

EMBEDDING_FILES = [
    'crawl-300d-2M.vec',
    'glove.840B.300d.txt'
]

NUM_MODELS = 5
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


def bi_gru_model(input_shape, embedding_matrix, gru_units, num_aux_targets):
    inp = layers.Input(shape=input_shape)
    x = layers.Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(inp)
    x = layers.SpatialDropout1D(0.3)(x)
    x = layers.Bidirectional(layers.CuDNNGRU(gru_units, return_sequences=True))(x)
    x = layers.Bidirectional(layers.CuDNNGRU(gru_units, return_sequences=True))(x)

    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    conc = layers.concatenate([avg_pool, max_pool])

    out_1 = layers.Dense(1, activation='sigmoid')(conc)
    out_2 = layers.Dense(num_aux_targets, activation='sigmoid')(conc)
    model = Model(inputs=inp, outputs=[out_1, out_2])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


def cnn_model(input_shape, embedding_matrix, num_aux_targets, filter_sizes=[2, 3, 5], num_filters=256):
    inp = layers.Input(shape=input_shape)
    x = layers.Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(inp)
    reshape = layers.Reshape((input_shape[0], embedding_matrix.shape[1], 1))(x)
    conv_1 = layers.Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_matrix.shape[1]), padding='valid',
                           kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = layers.Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_matrix.shape[1]), padding='valid',
                           kernel_initializer='normal', activation='relu')(reshape)
    conv_3 = layers.Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_matrix.shape[1]), padding='valid',
                           kernel_initializer='normal', activation='relu')(reshape)
    pool_1 = layers.MaxPool2D(pool_size=(input_shape[0] - filter_sizes[0] + 1, 1), strides=(1, 1))(conv_1)
    pool_2 = layers.MaxPool2D(pool_size=(input_shape[0] - filter_sizes[1] + 1, 1), strides=(1, 1))(conv_2)
    pool_3 = layers.MaxPool2D(pool_size=(input_shape[0] - filter_sizes[2] + 1, 1), strides=(1, 1))(conv_3)
    concate = layers.concatenate([pool_1, pool_2, pool_3])
    flatten = layers.Flatten()(concate)
    dropout = layers.Dropout(0.3)(flatten)
    out_1 = layers.Dense(1, activation='sigmoid')(dropout)
    out_2 = layers.Dense(num_aux_targets, activation='sigmoid')(dropout)
    model = Model(inputs=inp, outputs=[out_1, out_2])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


def rnn_cnn_model(input_shape, embedding_matrix, gru_units, num_aux_targets):
    inp = layers.Input(shape=input_shape)
    x = layers.Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(inp)
    x = layers.SpatialDropout1D(0.3)(x)
    x = layers.Bidirectional(layers.CuDNNGRU(gru_units, return_sequences=True))(x)
    x = layers.Bidirectional(layers.CuDNNGRU(gru_units, return_sequences=True))(x)
    x = layers.Conv1D(64, kernel_size=2, kernel_initializer='he_uniform')(x)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    conc = layers.concatenate([avg_pool, max_pool])

    out_1 = layers.Dense(1, activation='sigmoid')(conc)
    out_2 = layers.Dense(num_aux_targets, activation='sigmoid')(conc)
    model = Model(inputs=inp, outputs=[out_1, out_2])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


class Attention(layers.Layer):
    def __init__(self, step_dim, bias=True, **kwargs):
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(None)
        self.b_regularizer = regularizers.get(None)
        self.W_constraint = constraints.get(None)
        self.b_constraint = constraints.get(None)
        self.bias = bias
        self.step_dim = step_dim
        self.feature_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(int(input_shape[-1]),), initializer=self.init, name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer, constraint=self.W_constraint)
        self.feature_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    def call(self, inputs, **kwargs):
        features_dim = self.feature_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(inputs, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))),
                        (-1, step_dim))
        if self.bias:
            eij += self.b

        eij = K.tanh(eij)
        a = K.exp(eij)

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = inputs * a
        return K.sum(weighted_input, axis=1)


def cnn_attention_model(input_shape, embedding_matrix, num_aux_targets, conv_units=128):
    inp = layers.Input(shape=input_shape)
    x = layers.Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(inp)
    x = layers.SpatialDropout1D(0.3)(x)
    att = Attention(input_shape[0])(x)

    x = layers.Conv1D(conv_units, 2, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(5, padding='same')(x)

    x = layers.Conv1D(conv_units, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(5, padding='same')(x)

    x = layers.Flatten()(x)
    x = layers.concatenate([x, att])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)

    out_1 = layers.Dense(1, activation='sigmoid')(x)
    out_2 = layers.Dense(num_aux_targets, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=[out_1, out_2])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

x_train = preprocess(train['comment_text'])[:10000]
y_train = np.where(train['target'] >= 0.5, 1, 0)[:10000]
y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']][:10000]
x_test = preprocess(test['comment_text'])

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(x_train) + list(x_test))
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

embedding_matrix = np.concatenate([build_matrix(tokenizer.index_word, f) for f in EMBEDDING_FILES], axis=-1)


def get_models():
    models = {}
    models['bi_lstm_model'] = bi_lstm_model(input_shape=(MAX_LEN,), embedding_matrix=embedding_matrix,
                                            lstm_units=LSTM_UNITS,
                                            num_aux_targets=y_aux_train.shape[-1])
    models['bi_gru_model'] = bi_gru_model(input_shape=(MAX_LEN,), embedding_matrix=embedding_matrix, gru_units=128,
                                          num_aux_targets=y_aux_train.shape[-1])
    models['cnn_model'] = cnn_model(input_shape=(MAX_LEN,), embedding_matrix=embedding_matrix,
                                    num_aux_targets=y_aux_train.shape[-1])
    models['rnn_cnn_model'] = rnn_cnn_model(input_shape=(MAX_LEN,), embedding_matrix=embedding_matrix, gru_units=128,
                                            num_aux_targets=y_aux_train.shape[-1])
    models['cnn_attention_model'] = cnn_attention_model(input_shape=(MAX_LEN,), embedding_matrix=embedding_matrix,
                                                        num_aux_targets=y_aux_train.shape[-1])
    return models


models = get_models()
ens_prediction = 0.0
for model_name, model in models.items():
    checkpoint_predictions = []
    weights = []
    for epoch_idx in range(EPOCHS):
        model.fit(x_train, [y_train, y_aux_train], batch_size=BATCH_SIZE, verbose=1,
                  callbacks=[LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** epoch_idx))])
        checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())
        weights.append(2 ** epoch_idx)
    model_prediction = np.average(checkpoint_predictions, weights=weights, axis=0)
    print(model_prediction)
    ens_prediction += model_prediction

ens_prediction /= len(models)

submission = pd.DataFrame.from_dict({
    'id': test['id'],
    'prediction': ens_prediction
})
submission.to_csv('submission.csv', index=False)
