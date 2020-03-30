from gensim.models import KeyedVectors
import numpy as np
import sys
import tqdm
import copy

model = KeyedVectors.load_word2vec_format(sys.argv[1])

with open(sys.argv[2], 'w', encoding='utf-8') as fout1:
    with open(sys.argv[3], 'w', encoding='utf-8') as fout2:
        for idx in tqdm.tqdm(range(len(model.vocab))):
            word = model.index2word[idx]
            most_similar = model.most_similar(word, topn=None)
            most_similar_index = np.argsort(-most_similar)
            most_similar_words = [model.index2word[widx] for widx in list(most_similar_index)]
            most_similar_index = list(most_similar_index)

            fout1.write(' '.join(most_similar_words)+'\n')
            fout2.write(' '.join([str(item) for item in most_similar_index])+'\n')