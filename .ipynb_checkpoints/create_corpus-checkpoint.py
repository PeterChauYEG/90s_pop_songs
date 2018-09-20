import csv
import numpy as np
import pandas as pd
from random import sample

# hyper parameters
max_char_n = 20
n_examples = 100000

# output
output_dir = 'data/'
charset_file = '{}charset.csv'.format(output_dir)
dataset_file = '{}dataset.csv'.format(output_dir)

# Functions
# Transforms a csv into an array of song lyrics (None, 1)
def csvToSongLyricsArray(csv):
    # filter for only lyrics from the 1990s, of the pop genre, and not instrumentals
    mask = (csv['year'] > 1989) & (csv['year'] < 2000) & (csv['genre'] == 'Pop') & (csv['lyrics'] != '[Instrumental]')
    filtered = csv[mask]
    
    # remove null values
    nonNull = filtered.dropna()
    
    # trim all the extra data. We only want the lyrics
    lyrics = nonNull['lyrics']
    
    # reindex the lyrics to make it easier to work with
    reindexed = lyrics.reset_index(drop=True)
    
    # lowercase the lyrics
    lowercased = reindexed[:].str.lower()
    
    # get the number of song lyrics
    n_songs = lowercased.shape[0]
    
    return lowercased, n_songs
    
# filter out any song where lyrics contain a character outside the chars set
def filterLyrics(charset, lyrics):
    filtered_lyrics = []
    
    # for each song
    for lyric in lyrics:
        check = 0
        
        # split the lyric into an array of characters
        lyric_chars = list(lyric)
        
        # for each character, check if it's not in the chars set
        for char in lyric_chars:
            if char not in charset:
                check = 1

        # if all character are in the chars set
        # add it to our filter lyrics list
        if check == 0:
            filtered_lyrics.append(lyric_chars)
            
    # get the number of lyrics
    n_filtered_lyrics = len(filtered_lyrics)    
    
    return filtered_lyrics, n_filtered_lyrics
    
# flatten the previous into a list of song lyrics lines
def flatten_lyrics(lyrics):
    flattened_lyrics = [line for song in lyrics for line in song]
    n_chars = len(flattened_lyrics)
    
    return flattened_lyrics, n_chars

def generateDataset(lyrics, n_chars, n_examples, max_char_n):
    dataset = []
    max_index = n_chars - max_char_n
    start_indices = np.random.randint(0, max_index, size=n_examples)

    for start_index in start_indices:
        end_index = start_index + max_char_n
        example = lyrics[start_index:end_index]
        start_index = end_index
        dataset.append(example)
        
    return dataset
    
# Extract and Transform Raw Dataset    
# load raw data file as a dataframe
raw_data = pd.read_csv('data/raw.csv')

# get formatted_lyrics and number of songs
lyrics, n_lyrics = csvToSongLyricsArray(raw_data)

lyrics.head(10)

# examine the number of song lyrics we have
print("Number of lyrics: {}".format(n_lyrics))
print("Lyric Example: {}".format(lyrics[0]))

# Filter out non-english lyrics
charset = ["'", 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'x', 'z', '\n', '!', '"', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ']

# determine number of charecters in our set
n_charset = len(charset)

print("Number of characters in chars: {}".format(n_charset))

# filter out any song where lyrics contain a character outside the english set
filtered_lyrics, n_filtered_lyrics = filterLyrics(charset, lyrics)

print("Number of english songs: {}".format(n_filtered_lyrics))
print("A english song lyric: {}".format(filtered_lyrics[0]))

# flatten english song lyrics
flattened_lyrics, n_chars = flatten_lyrics(filtered_lyrics)

print("Number of song lyrics characters: {}".format(n_chars))
print("Section of song lyrics: {}".format(flattened_lyrics[0:100]))

# Extract the subset we are interested in
# generate n_examples example of max_char_n length
dataset = generateDataset(flattened_lyrics, n_chars, n_examples, max_char_n)

print("Number of examples in dataset: {}".format(len(dataset)))
print("Example: {}".format(dataset[0]))

# Export datasets
# save charset
with open(charset_file, 'w', newline='') as csvFile:
    file = csv.writer(csvFile, delimiter=',')
    file.writerows(charset)
    
# save dataset
with open(dataset_file, 'w', newline='') as csvFile:
    file = csv.writer(csvFile, delimiter=',')
