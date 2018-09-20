# Model Chaaracter Level Script

import csv
from coremltools import converters
from datetime import datetime
from IPython.display import SVG
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Dense, Input, LSTM
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
import numpy as np
import os
import sys

# hyper parameters
activations = 128
batch_size = 50
epochs = 10
learning_rate = 0.01
training_ratio = 0.3

# output
output_dir = 'data/'
charset_file = '{}charset.csv'.format(output_dir)
dataset_file = '{}dataset.csv'.format(output_dir)

# to use GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# verify that a gpu is listed
K.tensorflow_backend._get_available_gpus()

# functions
def deprocessPrediction(ix_to_char, prediction):
    index = np.argmax(prediction)
    char = ix_to_char[index]
    
    return char
    
def generateCharacterConverters(chars):
    char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
    ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }
    
    return char_to_ix, ix_to_char
    
def generateXYDatasets(char_to_ix, dataset, n_examples, n_charset, max_char_n):
    # create training input
    x_dataset = np.zeros((n_examples, max_char_n-1, n_charset), dtype='float32')
    
    # create training input
    y_dataset = np.zeros((n_examples, n_charset), dtype='float32')
    
    # fill input training set with word sequences, where words are one-hot encoded
    for li, line in enumerate(dataset):
        for ci, char in enumerate(line[:-1]):
            index = char_to_ix[char]
            x_dataset[li][ci][index] = 1
            
    # create training output
    for li, line in enumerate(dataset):
        char = line[-1]
        index = char_to_ix[char]
        y_dataset[li][index] = 1
        
    return x_dataset, y_dataset
    
def preprocessExample(char_to_ix, example, n_chars_set):
    chars = list(example)
    n_sample_chars = len(chars)

    preprocessed_example = np.zeros((1, n_sample_chars, n_chars_set), dtype='float32')

    for ci, char in enumerate(chars):
        index = char_to_ix[char]
        preprocessed_example[0][ci][index] = 1

    return preprocessed_example
    
def sample_predictions(preds, temperature=0.5):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return probas
    
# Load data    
with open(charset_file, 'r') as csv_file:
    reader = csv.reader(csv_file, delimiter=",")
    charset = []
    for row in reader:
        charset.append(row[0])

with open(dataset_file, 'r') as csv_file:
    reader = csv.reader(csv_file, delimiter=",")
    dataset = []
    for row in reader:
        dataset.append(row)
        
# Generate Dataset        
# create dictionarys
char_to_ix, ix_to_char = generateCharacterConverters(charset)
n_charset = len(charset)

print("Number of characters: {}".format(n_charset))
print("ix_to_char: {}".format(ix_to_char))
print("char_to_ix: {}".format(char_to_ix))

# create training input and output
n_examples = len(dataset)
max_char_n = len(dataset[0])
x_dataset, y_dataset = generateXYDatasets(char_to_ix, dataset, n_examples, n_charset, max_char_n)
print("Number of examples: {}".format(n_examples))
print("max characters: {}".format(max_char_n))

# Validate Dataset
x_example = x_dataset[2] 

x_example_string = []
for woh in x_example:
    char = deprocessPrediction(ix_to_char, woh)
    x_example_string.append(char)
x_example_string_formatted = "".join(x_example_string)

print("x_example shape: {}".format(x_example.shape))
print("x_example one-hot: {}".format(x_example))
print("x_example: {}".format(x_example_string_formatted))

y_example = y_dataset[2]

char = deprocessPrediction(ix_to_char, y_example)

print("y_example shape: {}".format(y_example.shape))
print("y_example one-hot: {}".format(y_example))
print("y_example: {}".format(char))

# Model
model_input = Input(shape=(None, n_charset))
x = LSTM(activations)(model_input)
x = Dense(n_charset, activation='softmax')(x)

model = Model(inputs=model_input, outputs=x)

optimizer = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Plot model
# generate logging variables
variant = '{}ex-{}a-{}b-{}c'.format(n_examples, activations, batch_size, max_char_n)
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
diagram_dir = 'diagram/{}-{}'.format(variant, timestamp)
log_dir = 'logs/{}-{}'.format(variant, timestamp)
mlmodel_dir = 'model/{}-{}.mlmodel'.format(variant, timestamp)
model_dir = 'model/{}-{}.hdf5'.format(variant, timestamp)

# draw a model diagram and save it to disk
plot_model(model, to_file=diagram_dir)
SVG(model_to_dot(model).create(prog='dot', format='svg'))

model.summary()

# set up callbacks
early = EarlyStopping(monitor='val_acc',
                      min_delta=0,
                      patience=10,
                      verbose=1,
                      mode='auto')
                      
model.fit(x_dataset, 
          y_dataset, 
          batch_size=batch_size, 
          epochs=epochs, 
          shuffle=True,
          validation_split=training_ratio,
          callbacks=[early, TensorBoard(log_dir=log_dir)])                      

# Export Model
model.save(model_dir)

# Export CoreML Model
coreml_model = converters.keras.convert(model)
coreml_model.save(mlmodel_dir)