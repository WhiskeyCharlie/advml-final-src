from gensim import models

DATA_PATH = '/home/jon/Documents/AdvML/final_project/data/google/GoogleNews-vectors-negative300.bin'

w = models.KeyedVectors.load_word2vec_format(DATA_PATH, binary=True)
for vec in w:
    print(vec)
    break
