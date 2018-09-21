import csv
from keras import backend as K
from keras.models import load_model
import numpy as np
import os
import sys

# inputs
model_input = sys.argv[1]
n_chars = int(sys.argv[2])
model_dir = sys.argv[3]
charset_file = sys.argv[4]

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

def preprocessInput(char_to_ix, input, n_chars_set):
    chars = list(input)
    n_sample_chars = len(chars)

    preprocessed_input = np.zeros((1, n_sample_chars, n_chars_set), dtype='float32')

    for ci, char in enumerate(chars):
        index = char_to_ix[char]
        preprocessed_input[0][ci][index] = 1

    return preprocessed_input
    
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
char_to_ix, ix_to_char = generateCharacterConverters(charset)
n_charset = len(charset)

# Load Model
model = load_model(model_dir)

# Generate a sequence from a sequence
# convert input to a sequence of one-hot encoded chars
preprocessed_input = preprocessInput(char_to_ix, model_input, n_charset)

result = model_input

for i in range(n_chars):
    prediction = model.predict(preprocessed_input, verbose=0)[0]
    sampled_prediction = sample_predictions(prediction)
    next_char = deprocessPrediction(ix_to_char, sampled_prediction[0])
    preprocessed_input[0][:-1] = preprocessed_input[0][1:]
    preprocessed_input[0][-1] = sampled_prediction
    result += next_char

sys.stdout.write("\nRAD NEW 90s POP LYRICS")
sys.stdout.write("\n======================")
sys.stdout.write("\n\n")
sys.stdout.write(result)
sys.stdout.write("\n")
sys.stdout.write("\n======================\n")