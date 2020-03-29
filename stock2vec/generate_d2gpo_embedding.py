import sys
from gensim.models.wrappers import FastText
from gensim.scripts.glove2word2vec import glove2word2vec
model = FastText.load_fasttext_format(sys.argv[1])

with open(sys.argv[2], 'r', encoding='utf-8') as fin:
    with open(sys.argv[3], 'w', encoding='utf-8') as fout:
        for line in fin:
            word = line.strip()
            if word in model:
                embs = list(model[word])
            else:
                embs = [1e-8 for _ in range(model.vector_size)]
            embs = [str(item) for item in embs]
            fout.write(' '.join([word]+embs)+'\n')

if len(sys.argv)>4 is not None:
    glove2word2vec(sys.argv[3], sys.argv[4])