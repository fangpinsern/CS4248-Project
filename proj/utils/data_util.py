import torchtext
import torch
import numpy as np
import os 

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from ..constants import JSON_FILE, TRAIN_TEST_SPLIT_FILE, CATEGORY_SUBSET

# DATASET
# =========================================================================
def get_dataset():
    a = pd.read_json(JSON_FILE, lines=True)
    a = a.drop(a[a['headline']==''].index.values)
    a = a[a['category'].isin(CATEGORY_SUBSET)]
    return a[['headline', 'category']]

def train_test_split(percent_train=0.7, percent_dev=0.1, percent_test=0.2):
    if os.path.exists(TRAIN_TEST_SPLIT_FILE):
        data = pd.read_csv(TRAIN_TEST_SPLIT_FILE)
        return data

    data = get_dataset()
    l = len(data)
    train_num = int(l*0.7)
    dev_num = int(l*0.1)
    rnd_ind = np.arange(l)
    np.random.shuffle(rnd_ind)
    train_ind = rnd_ind[:train_num]
    dev_ind = rnd_ind[train_num:train_num+dev_num]
    test_ind = rnd_ind[train_num+dev_num:]
    data = data.reset_index()
    data['ind'] = data.index

    def change_phase(ind):
        if ind in train_ind:
            return 'train'
        elif ind in dev_ind:
            return 'dev'
        else:
            return 'test'

    data['phase'] = data['ind'].apply(change_phase)
    data = data.drop(columns=['ind'])

    data.to_csv(TRAIN_TEST_SPLIT_FILE, index=False)

    return data
    

# EMBEDDINGS
# =========================================================================

# input: 
    # glove_path: path to your 6B 50d glove file; if no glove_path input, then will take around 5m to download the glove_embeddings
    # output: can be Tensor or Array; Array for no glove_path input is not implemented
# return: 
    # a dictionary (or a pytorch vocab object) with {word: 50 dimensional embedding vector}
def get_glove_embeddings(glove_path=None, output='Tensor'):
    if glove_path is not None:
        glove = {}
        with open(glove_path, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                if output == 'Tensor':
                    glove[word] = torch.from_numpy(vector)
                else:
                    glove[word] = vector
        return glove

    glove = torchtext.vocab.GloVe(name="6B", dim=50)

    if output != 'Tensor':
        print("not available! if want embeddings in array, input the glove_path!")
        return None
      
    return glove
    
def get_stopwords():
    nltk.download('stopwords')
    STOPWORDS = stopwords.words("english")
    STOPWORDS = set(STOPWORDS) | set(ENGLISH_STOP_WORDS)
    return STOPWORDS