import torchtext
import torch
import numpy as np
import os 
import pandas as pd

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from ..constants import JSON_FILE, TRAIN_TEST_SPLIT_FILE, CATEGORY_SUBSET, BIGRAM_TRIGRAM_VOCAB

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
    
def balanced_train_test_split(percent_train=0.7, percent_dev=0.1, percent_test=0.2, count=2000):
    def balance_train_data(train_data, count):
        ret = None
        for cat in CATEGORY_SUBSET:
            data_of_cat = data[data['category']==cat]
            data_of_cat = data_of_cat.sample(count, replace=True)
            if ret is None:
                ret = data_of_cat
            else:
                ret = pd.concat([ret, data_of_cat], axis=0)
        return ret
      
    if os.path.exists(TRAIN_TEST_SPLIT_FILE):
        data = pd.read_csv(TRAIN_TEST_SPLIT_FILE)
        train_data = data[data['phase']=='train']
        other_data = data[data['phase']!='train']
        train_data = balance_train_data(train_data, count)
        data = pd.concat([train_data, other_data], axis=0)
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

    train_data = data[data['phase']=='train']
    other_data = data[data['phase']!='train']
    train_data = balance_train_data(train_data, count)
    data = pd.concat([train_data, other_data], axis=0)

    return data
    
# STOPWORDS
# =========================================================================

def get_top_n_ngrams(corpus, ngram_range, n=20, remove_stopwords=False):
    vec = None
    if remove_stopwords:
        vec = CountVectorizer(ngram_range=ngram_range, stop_words = 'english').fit(corpus)
    else:
        vec = CountVectorizer(ngram_range=ngram_range).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def get_frequent_ngrams_for_categories(dataset, ngram_range, n=20, remove_stopwords=False):
    vocab = {}
    for i in range(len(CATEGORY_SUBSET)):
        cat = CATEGORY_SUBSET[i]
#         print(f"{cat}: ")   
        common_words = get_top_n_ngrams(dataset[dataset['category']==cat]['headline'], ngram_range, n, remove_stopwords=remove_stopwords)
        for word, freq in common_words:
#             print(f"    {word}, {freq}")
            r_freq = freq / len(dataset[dataset['category']==cat]['headline'])
            if word not in vocab:
                vocab[word] = [r_freq if j == i else 0 for j in range(len(CATEGORY_SUBSET))]
            else:
                vocab[word][i] = r_freq
    df = pd.DataFrame(vocab)
    df.index = CATEGORY_SUBSET
    df = df.T
    df = df.sort_values(by=CATEGORY_SUBSET)
    return df
    
def get_stopwords(dataset=None, include_common_unigram=False, uninclude_certain_unigram=False):
    nltk.download('stopwords')
    STOPWORDS = stopwords.words("english")
    STOPWORDS = set(STOPWORDS) | set(ENGLISH_STOP_WORDS)
    
    if dataset is None:
        dataset = get_dataset()
        
    if include_common_unigram:
        df = get_frequent_ngrams_for_categories(dataset, (1,1), n=20, remove_stopwords=True)
        df[df != 0] = 1
        df['sum'] = df.sum(axis=1)
        common_unigrams = df[df['sum'] > 3].index.tolist()
        STOPWORDS = STOPWORDS.union(common_unigrams)
        
    if uninclude_certain_unigram:
        stopwords_to_keep = ['ever', 'll', 'best', 'third', 'your', 'against', 'had', 'former', 'he', 'her', 'his', 'video', 'found', 'after']
        for s in stopwords_to_keep:
        try:
            STOPWORDS.remove(s)
        except:
            continue
            
    return STOPWORDS
    
# TOKENIZATION
# =========================================================================

def break_hashtag(text):
    if re.match(r'#\w+', text):
        words = []
        i = 1
        word = ''
        while i < len(text):
            if text[i].isupper():
                words.append(word)
                word = text[i]
            else:
                word += text[i]
            i += 1
        words.append(word)
        return ' '.join(words).strip()
    else:
        return text
        
def tokenize(text, with_stopwords=False):
    text = break_hashtag(text)
    text = re.sub(r'[^\w]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [t.lower() for t in tokens]
    
    lem = WordNetLemmatizer()
    if with_stopwords:
        s_tokens = [t for t in tokens if re.match(r"\w+", t) and t not in STOPWORDS]
        s_tokens = [lem.lemmatize(t) for t in s_tokens]
        if len(s_tokens) > 0:
            return s_tokens
    
    # return [lem.lemmatize(t) for t in tokens]
    return [t for t in tokens]
    
    
class Bigram_Trigram_Tokenizer:
    def __init__(self):
        self.bigram_trigram_vocab = pd.read_csv(BIGRAM_TRIGRAM_VOCAB)
        
    def get_PMI_for_word(self, word):
        pmi = self.bigram_trigram_vocab[self.bigram_trigram_vocab['ngram']==word]['PMI'].values
        if len(pmi) == 0:
            return 0

        return pmi[0]

    def tokenize_with_bigrams(self, text):
        unigrams = tokenize(text)
        bigrams = [' '.join(t) for t in list(zip(unigrams, unigrams[1:]+[" "]))]
        bigrams_pmi = [self.get_PMI_for_word(word) for word in bigrams]

        def helper(left_start, right_end):    
            if left_start >= right_end:
            return ''

        max_bigram_arg = np.argmax(bigrams_pmi[left_start:right_end]) + left_start
        if bigrams_pmi[max_bigram_arg] > 0:
            left = helper(left_start, max_bigram_arg)
            right = helper(max_bigram_arg+2, right_end)
            bi_unigram = '_'.join(bigrams[max_bigram_arg].split(' '))
            return ' '.join([left, bi_unigram, right])
        else:
            return ' '.join(unigrams[left_start:right_end])

        ret = helper(0, len(unigrams))
        return nltk.word_tokenize(ret)
        
    def get_bigram_trigram_token_list(self):
        return self.bigram_trigram_vocab['token'].values

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
    