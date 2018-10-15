# 90s Pop Lyrics Generator

This repository is an exploration on sequence-to-sequence models using RNNs. It attempts to build an AI which can generate 90s pop lyrics.

Details can be found with it's accompanying [blog post](http://labone.tech/90s-pop-lyrics-generator/).

## Usage
1. Obtain the required [dataset](https://www.kaggle.com/gyani95/380000-lyrics-from-metrolyrics)
2. Place it at `data/dataset.csv`
3. Run the `create_corpus.py` to generate the corpus.
4. Run the `model.py` to train the model.
5. Run the `generate_lyrics.py "some input" 500 'model/100000ex-128a-50b-20c-2018-09-19 19:41:58.hdf5' 'data/charset.csv'` generate lyrics.
    - "some input" being the starting string
    - 500 being the number of chars to generate
    - 'model/100000ex-128a-50b-20c-2018-09-19 19:41:58.hdf5' being the model file
    - 'data/charset.csv' being the charset array file

peace. love. spice up your life.
