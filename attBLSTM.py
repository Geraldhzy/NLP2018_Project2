import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keras import regularizers
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

from gensim.models import KeyedVectors
print('reading embedding...')
word_vectors = KeyedVectors.load_word2vec_format('word2vec_300d.txt', binary=False)

import sys
train_data_file = sys.argv[1]
test_data_file = sys.argv[2]
result_path = sys.argv[3]

def clear_puncts(data):
    data = re.sub("<e1>|</e1>|<e2>|</e2>", "", data)
    data = re.sub("[0-9\<\>\.\!\/_,~@#$&%^*\:\?()\+\-\=\"\']", " ", data.lower())
    return data

def tokenize(data):
    data = word_tokenize(data)
    return data

def preprocess(data):
    data = clear_puncts(data)
    data = tokenize(data)
    return data
print('processing data...')
test_data = {}
with open (test_data_file) as f:
    for line in f:
        l = line.split('\t')
        idx = l[0]
        test_data[idx] = {}
        text = l[1]
        e1_big = text.find('<e1>')
        e1_end = text.find('</e1>')
        e2_big = text.find('<e2>')
        e2_end = text.find('</e2>')
        e1 = text[e1_big + 4: e1_end]
        e2 = text[e2_big + 4: e2_end]
        test_data[idx]['text'] = preprocess(text)
        test_data[idx]['e1'] = e1
        test_data[idx]['e2'] = e2

train_data = {}
with open(train_data_file) as f:
    for i in range(8000):
        line = f.readline()
        relation = f.readline().strip('\n')
        comment = f.readline()
        f.readline()
        l = line.split('\t')
        idx = l[0]
        text = l[1]
        train_data[idx] = {}
        e1_big = text.find('<e1>')
        e1_end = text.find('</e1>')
        e2_big = text.find('<e2>')
        e2_end = text.find('</e2>')
        e1 = text[e1_big + 4: e1_end]
        e2 = text[e2_big + 4: e2_end]
        train_data[idx]['text'] = preprocess(text)
        train_data[idx]['e1'] = e1
        train_data[idx]['e2'] = e2
        h = relation.find('(')
        t = relation.find(',')
        train_data[idx]['rel'] = relation[:h]
        train_data[idx]['head'] = relation[h + 1:t]
        train_data[idx]['tail'] = relation[t + 1:-1]

classes = {'Cause-Effect':{'e1':0, 'e2':1}, \
           'Instrument-Agency':{'e1':2, 'e2':3}, \
           'Product-Producer':{'e1':4, 'e2':5}, \
           'Content-Container':{'e1':6, 'e2':7}, \
           'Entity-Origin':{'e1':8, 'e2':9}, \
           'Entity-Destination':{'e1':10, 'e2':11},
           'Component-Whole':{'e1':12, 'e2':13},
           'Member-Collection':{'e1':14, 'e2':15},
           'Message-Topic':{'e1':16, 'e2':17},
           'Othe':18}
inf_classes = {0:'Cause-Effect(e1,e2)', 1:'Cause-Effect(e2,e1)',
                2:'Instrument-Agency(e1,e2)',3:'Instrument-Agency(e2,e1)', \
                4:'Product-Producer(e1,e2)', 5:'Product-Producer(e2,e1)', \
                6:'Content-Container(e1,e2)', 7:'Content-Container(e2,e1)', \
                8:'Entity-Origin(e1,e2)', 9:'Entity-Origin(e2,e1)',\
                10:'Entity-Destination(e1,e2)', 11:'Entity-Destination(e2,e1)', \
                12:'Component-Whole(e1,e2)', 13:'Component-Whole(e2,e1)', \
                14:'Member-Collection(e1,e2)', 15:'Member-Collection(e2,e1)', \
                16:'Message-Topic(e1,e2)', 17:'Message-Topic(e2,e1)', 18:'Other'}
max_length = 86

train_x = []
train_y = []
for d in train_data:
    rel = train_data[d]['rel']
    y = [0]*19
    if rel == 'Othe':
        y[18] = 1
    else:
        y[classes[rel][train_data[d]['head']]] = 1
    train_y.append(y)
    s = []
    for w in train_data[d]['text']:
        if w in word_vectors:
            s.append(word_vectors[w])
        else:
            s.append(word_vectors['UNK'])
    while len(s) < 86:
        s.append(np.zeros(300))
    train_x.append(s)

X_val = train_x[:800]
Y_val = train_y[:800]
X = train_x[800:]
Y = train_y[800:]

test_x = []
for d in test_data:
    s = []
    for w in test_data[d]['text']:
        if w in word_vectors:
            s.append(word_vectors[w])
        else:
            s.append(word_vectors['UNK'])
    while len(s) < 86:
        s.append(np.zeros(300))
    test_x.append(s)

print('compiling model...')

def attention_3d_block(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(86, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

inputs = Input(shape=(86, 300))
drop1 = Dropout(0.3)(inputs)
lstm_out = Bidirectional(LSTM(units=128, unit_forget_bias=True, implementation=2, activation='tanh', recurrent_activation='hard_sigmoid',kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', return_sequences=True), name='bilstm')(drop1)
attention_mul = attention_3d_block(lstm_out)
attention_flatten = Flatten()(attention_mul)
drop2 = Dropout(0.3)(attention_flatten)
output = Dense(19, activation='softmax')(drop2)
model = Model(inputs=inputs, outputs=output)
model.compile('RMSprop', 'categorical_crossentropy', metrics=['accuracy'])

model.summary()

nb_epoch = 30
batch_size = 256
save_path = 'attBLSTM.h5'
checkpoint = ModelCheckpoint(filepath=save_path,
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     monitor='val_acc',
                                     mode='max' )
csv_logger = CSVLogger('%s-log.csv'%'attBLSTM', separator=',', append=False)
earlystopping = EarlyStopping(monitor='val_acc', patience = 4, verbose=1, mode='max')
history = model.fit(X, Y,
                      validation_data=(X_val, Y_val),
                      epochs=nb_epoch,
                      batch_size=batch_size,
                      callbacks=[checkpoint, earlystopping, csv_logger])

model = load_model('attBLSTM.h5')
y_pred = model.predict(test_x)
result = []
for y in y_pred:
    result.append(np.argmax(y))

with open(result_path, 'w') as f:
    for i,y in enumerate(result):
        f.write(str(i+8001) + '\t' + inf_classes[y] + '\n')
