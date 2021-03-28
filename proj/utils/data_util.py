import torchtext
import torch
import numpy as np
import os 

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