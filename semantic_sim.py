import collections

import numpy as np
from scipy import spatial

# import gensim
# from gensim.models import Doc2Vec
# from gensim.models.doc2vec import TaggedDocument  # ,LabeledSentence
# import random
import lexical_syntactic as lex_s


def semantic_composition(df_, word_idf, word_tf, nlp):  # , sent)
    #   tok_lis = []
    #   for s in sent:
    #       tok_lis.append(lex_s.tokenize(s))

    #   model = gensim.models.Word2Vec(tok_lis, size=100, window=5, min_count=1, workers=4, iter=12, seed=1)

    cos_word2vec = []s

    for ind, row in df_.iterrows():
        s1_wordtfidf = word_idf.fromkeys(word_idf, 0)
        s2_wordtfidf = word_idf.fromkeys(word_idf, 0)

        s1 = lex_s.tokenize(row['Sent1'].decode('utf-8').encode('ascii', 'ignore').strip())
        s2 = lex_s.tokenize(row['Sent2'].decode('utf-8').encode('ascii', 'ignore').strip())

        doc1 = nlp(row['Sent1'].decode('utf-8').lower())
        doc2 = nlp(row['Sent2'].decode('utf-8').lower())

        word_vec1 = collections.defaultdict(lambda: 0)
        for tok in doc1:
            word_vec1[tok.text] = tok.vector

        word_vec2 = collections.defaultdict(lambda: 0)
        for tok in doc2:
            word_vec2[tok.text] = tok.vector

        for w in s1:
            s1_wordtfidf[w] = np.sum(word_tf[w]*word_idf[w]*word_vec1[w], axis=0)

        for w in s2:
            s2_wordtfidf[w] = np.sum(word_tf[w]*word_idf[w]*word_vec2[w], axis=0)

        # print("ind=" + str(ind))
        cos_word2vec.append(1 - spatial.distance.cosine(s1_wordtfidf.values(), s2_wordtfidf.values()))
        # print("ind=" + str(ind))

    df_['Semantic Compostion'] = cos_word2vec

    return df_


def para2vec(df_, nlp):  # , sent)
    #    documents = []
    #    ind = 0
    #    for s in sent:  # Sent is the corpus containing all the sentences present in the file.
    #        documents.append(TaggedDocument(lex_s.tokenize(s.decode('utf-8').encode('ascii', 'ignore').strip()),
    #                                        [str(ind)]))
    #        ind += 1
    #    random.seed(50)
    #    model = gensim.models.doc2vec.Doc2Vec(documents, size=50, min_count=2, iter=55, seed=1)
    # model.build_vocab(documents)
    # %time model.train(documents, total_examples=model.corpus_count, epochs=model.iter)

    doc_cos = []
    for ind, row in df_.iterrows():
        # arr1 = model.infer_vector(lex_s.tokenize(row['Sent1'].decode('utf-8').encode('ascii', 'ignore').strip()))
        # arr2 = model.infer_vector(lex_s.tokenize(row['Sent2'].decode('utf-8').encode('ascii', 'ignore').strip()))

        arr1 = nlp(row['Sent1'].decode('utf-8').lower())
        arr2 = nlp(row['Sent2'].decode('utf-8').lower())
        doc_cos.append(1 - spatial.distance.cosine(arr1.vector, arr2.vector))
        # print("ind="+str(ind))
    df_['Para2vec'] = doc_cos
    
    return df_
