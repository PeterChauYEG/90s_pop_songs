import csv
from keras import backend as K
from keras.models import load_model
import numpy as np
import os
import sys

# hyper parameters
activations = 128
example = 'sweet dreams are made of these'
n_chars = 500

# output
output_dir = 'data/'
charset_file = '{}charset.csv'.format(output_dir)
model_dir = 'model/100000ex-128a-50b-20c-2018-09-19 19:41:58.hdf5'

# to use GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# verify that a gpu is listed
K.tensorflow_backend._get_available_gpus()

# Functions
def deprocessPrediction(ix_to_char, prediction):
    index = np.argmax(prediction)
    char = ix_to_char[index]
    
    return char
    
def generateCharacterConverters(chars):
    char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
    ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }
    
    return char_to_ix, ix_to_char

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
    
# Load Data    
with open(charset_file, 'r') as csv_file:
    reader = csv.reader(csv_file, delimiter=",")
    charset = []
    for row in reader:
        charset.append(row[0])
        
# Generate Charset Dictionaries
# create dictionarys
char_to_ix, ix_to_char = generateCharacterConverters(charset)
n_charset = len(charset)

print("Number of characters: {}".format(n_charset))
print("ix_to_char: {}".format(ix_to_char))
print("char_to_ix: {}".format(char_to_ix))

# Load Model
model = load_model(model_dir)

# Generate a sequence from a sequence
# convert example to a sequence of one-hot encoded chars
preprocessed_example = preprocessExample(char_to_ix, example, n_charset)

sys.stdout.write(example)

for i in range(n_chars):
    prediction = model.predict(preprocessed_example, verbose=0)[0]
    sampled_prediction = sample_predictions(prediction)
    next_char = deprocessPrediction(ix_to_char, sampled_prediction[0])
    preprocessed_example[0][:-1] = preprocessed_example[0][1:]
    preprocessed_example[0][-1] = sampled_prediction
    sys.stdout.write(next_char)
    sys.stdout.flush()
print()        