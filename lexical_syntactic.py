import collections
import math
from difflib import SequenceMatcher
from string import punctuation

from nltk import SnowballStemmer
from nltk import ngrams
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy import spatial

stemmer = SnowballStemmer('english')
stop_words = stopwords.words('english') + list(punctuation)

##############################################################################################


def ngram(sentence, n):  # NGRAM maker

    wordnet_lemmatizer = WordNetLemmatizer()
    wt1 = word_tokenize(sentence)
    lemma = []
    for w in wt1:
        lemma.append(wordnet_lemmatizer.lemmatize(w))

    s = ' '.join(lemma)
    grams = ngrams(s.decode('utf_8').lower().split(), n)
    lis = []
    for gram in grams:
        lis.append(' '.join(gram))
        # print ' '.join(gram)
    return lis


def lcs(a, b):
    # Length of Longest common Subseq.
    lengths = [[0]*(len(b)+1)] * (len(a)+1)  # [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
    # read the substring out from the matrix
    result = ""
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x-1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y-1]:
            y -= 1
        else:
            assert a[x-1] == b[y-1]
            result = a[x-1] + result
            x -= 1
            y -= 1
    return result.split()


def inverse_document_frequencies(tokenized_documents, tokens):  # IDF
    idf_values = {}
    for tkn in tokens:
        s = 0
        for doc in tokenized_documents:
            if tkn.lower() in doc:
                s = s+1
        idf_values[tkn] = 1 + math.log((len(tokenized_documents)/s*1.0))
    return idf_values


def tfidf(tokenized_documents, tokens):  # Probability*IDF
    idf = inverse_document_frequencies(tokenized_documents, tokens)
    lis = [item for sublist in tokenized_documents for item in sublist]
    sc = 0
    for term in idf.keys():
        sc = sc + idf[term]*(lis.count(term.lower())*1.0/len(lis))  # IDF*Prob
    return sc


def nozip_ngram(text, n):  # Character Level n-gram
    return [text[i:i+n] for i in range(len(text)-n+1)]


def tokenize(text):  # Tokenization
    words = word_tokenize(text)
    words = [w.lower() for w in words]
    return [w for w in words if not w.isdigit()]


def lemma_ngram(df_, sent):
    one_gram = []  # Unigram tokenized list of list
    two_gram = []  # Bigram tokenized list of list
    three_gram = []  # Trigram tokenized list of list
    four_gram = []  # Quadgram tokenized list of list

    for s in sent:
        one_gram.append(ngram(s, n=1))
        two_gram.append(ngram(s, n=2))
        three_gram.append(ngram(s, n=3))
        four_gram.append(ngram(s, n=4))

    one = []
    two = []
    three = []
    four = []

    c_ab_one = []
    c_ab_two = []
    c_ab_three = []
    c_ab_four = []

    c_ba_one = []
    c_ba_two = []
    c_ba_three = []
    c_ba_four = []

    lcs1 = []

    weight_one = [] 
    weight_two = []
    weight_three = []
    weight_four = []

    for ind, row in df_.iterrows():

        # For 1-gram
        one_gram1 = ngram(row['Sent1'], n=1)
        one_gram2 = ngram(row['Sent2'], n=1)
        # For 2-gram
        two_gram1 = ngram(row['Sent1'], n=2)
        two_gram2 = ngram(row['Sent2'], n=2)
        # For 3-gram
        three_gram1 = ngram(row['Sent1'], n=3)
        three_gram2 = ngram(row['Sent2'], n=3)
        # For 4-gram
        four_gram1 = ngram(row['Sent1'], n=4)
        four_gram2 = ngram(row['Sent2'], n=4)

        # For Jaccard Coeff. and Containment coeff.
        token1 = set(one_gram1).intersection(one_gram2)
        token2 = set(two_gram1).intersection(two_gram2)
        token3 = set(three_gram1).intersection(three_gram2)
        token4 = set(four_gram1).intersection(four_gram2)

        # Jaccard Coeff.
        if len(set(one_gram1).union(one_gram2)) >= 1:
            one.append(len(token1)*1.0/len(set(one_gram1).union(one_gram2)))
        else:
            one.append(0)

        if len(set(two_gram1).union(two_gram2)) >= 1:
            two.append(len(token2)*1.0/len(set(two_gram1).union(two_gram2)))
        else:
            two.append(0)

        if len(set(three_gram1).union(three_gram2)) >= 1:
            three.append(len(token3)*1.0/len(set(three_gram1).union(three_gram2)))
        else:
            three.append(0)

        if len(set(four_gram1).union(four_gram2)) >= 1:
            four.append(len(token4)*1.0/len(set(four_gram1).union(four_gram2)))
        else:
            four.append(0)

        # Containment of A in B
        if len(one_gram1) >= 1:
            c_ab_one.append(len(token1)*1.0/len(set(one_gram1)))
        else:
            c_ab_one.append(0)

        if len(one_gram1) >= 2:
            c_ab_two.append(len(token2)*1.0/len(set(two_gram1)))
        else:
            c_ab_two.append(0)

        if len(one_gram1) >= 3:
            c_ab_three.append(len(token3)*1.0/len(set(three_gram1)))
        else:
            c_ab_three.append(0)

        if len(one_gram1) >= 4:
            c_ab_four.append(len(token4)*1.0/len(set(four_gram1)))
        else:
            c_ab_four.append(0)

        # Containment of B in A
        if len(one_gram2) >= 1:
            c_ba_one.append(len(token1)*1.0/len(set(one_gram2)))
        else:
            c_ba_one.append(0)

        if len(one_gram2) >= 2:
            c_ba_two.append(len(token2)*1.0/len(set(two_gram2)))
        else:
            c_ba_two.append(0)

        if len(one_gram2) >= 3:
            c_ba_three.append(len(token3)*1.0/len(set(three_gram2)))
        else:
            c_ba_three.append(0)

        if len(one_gram2) >= 4:
            c_ba_four.append(len(token4)*1.0/len(set(four_gram2)))
        else:
            c_ba_four.append(0)

        # Longest common sub-sequence divide by Union of one_gram1 and one_gram2
        lcs1.append(len(lcs(row['Sent1'], row['Sent2']))*1.0/len(set(one_gram1).union(one_gram2)))

        weight_one.append(tfidf(one_gram, token1))
        weight_two.append(tfidf(two_gram, token2))
        weight_three.append(tfidf(three_gram, token3))
        weight_four.append(tfidf(four_gram, token4))

    df_['Jaccard_1gram'] = one
    df_['Jaccard_2gram'] = two
    df_['Jaccard_3gram'] = three
    df_['Jaccard_4gram'] = four

    df_['CAB_1gram'] = c_ab_one
    df_['CAB_2gram'] = c_ab_two
    df_['CAB_3gram'] = c_ab_three
    df_['CAB_4gram'] = c_ab_four

    df_['CBA_1gram'] = c_ba_one
    df_['CBA_2gram'] = c_ba_two
    df_['CBA_3gram'] = c_ba_three
    df_['CBA_4gram'] = c_ba_four

    df_['LCSubseq'] = lcs1

    df_['ProbIDF_1gram'] = weight_one
    df_['ProbIDF_2gram'] = weight_two
    df_['ProbIDF_3gram'] = weight_three
    df_['ProbIDF_4gram'] = weight_four

    return df_


def pos_ngram(df_, nlp):
    one = []
    two = []
    three = []
    four = []

    c_ab_one = []
    c_ab_two = []
    c_ab_three = []
    c_ab_four = []

    c_ba_one = []
    c_ba_two = []
    c_ba_three = []
    c_ba_four = []

    for ind, row in df_.iterrows():

        doc1 = nlp(row['Sent1'].decode('utf-8'))
        doc2 = nlp(row['Sent2'].decode('utf-8'))

        p = [w.pos_ for w in doc1]
        q = [w.pos_ for w in doc2]

        s1 = []
        for a in p:
            s1.append(a)

        s1 = ' '.join(s1)

        s2 = []
        for a in q:
            s2.append(a)

        s2 = ' '.join(s2)

        one_gram1 = ngram(s1, n=1)
        one_gram2 = ngram(s2, n=1)
        # For 2-gram
        two_gram1 = ngram(s1, n=2)
        two_gram2 = ngram(s2, n=2)
        # For 3-gram
        three_gram1 = ngram(s1, n=3)
        three_gram2 = ngram(s2, n=3)
        # For 4-gram
        four_gram1 = ngram(s1, n=4)
        four_gram2 = ngram(s2, n=4)

        # For Jaccard Coeff. and Containment coeff.
        token1 = set(one_gram1).intersection(one_gram2)
        token2 = set(two_gram1).intersection(two_gram2)
        token3 = set(three_gram1).intersection(three_gram2)
        token4 = set(four_gram1).intersection(four_gram2)

        # Jaccard Coeff.
        if len(set(one_gram1).union(one_gram2)) >= 1:
            one.append(len(token1) * 1.0 / len(set(one_gram1).union(one_gram2)))
        else:
            one.append(0)

        if len(set(two_gram1).union(two_gram2)) >= 1:
            two.append(len(token2) * 1.0 / len(set(two_gram1).union(two_gram2)))
        else:
            two.append(0)

        if len(set(three_gram1).union(three_gram2)) >= 1:
            three.append(len(token3) * 1.0 / len(set(three_gram1).union(three_gram2)))
        else:
            three.append(0)

        if len(set(four_gram1).union(four_gram2)) >= 1:
            four.append(len(token4) * 1.0 / len(set(four_gram1).union(four_gram2)))
        else:
            four.append(0)

        # Containment of A in B
        if len(one_gram1) >= 1:
            c_ab_one.append(len(token1) * 1.0 / len(set(one_gram1)))
        else:
            c_ab_one.append(0)

        if len(one_gram1) >= 2:
            c_ab_two.append(len(token2) * 1.0 / len(set(two_gram1)))
        else:
            c_ab_two.append(0)

        if len(one_gram1) >= 3:
            c_ab_three.append(len(token3) * 1.0 / len(set(three_gram1)))
        else:
            c_ab_three.append(0)

        if len(one_gram1) >= 4:
            c_ab_four.append(len(token4) * 1.0 / len(set(four_gram1)))
        else:
            c_ab_four.append(0)

        # Containment of B in A
        if len(one_gram2) >= 1:
            c_ba_one.append(len(token1) * 1.0 / len(set(one_gram2)))
        else:
            c_ba_one.append(0)

        if len(one_gram2) >= 2:
            c_ba_two.append(len(token2) * 1.0 / len(set(two_gram2)))
        else:
            c_ba_two.append(0)

        if len(one_gram2) >= 3:
            c_ba_three.append(len(token3) * 1.0 / len(set(three_gram2)))
        else:
            c_ba_three.append(0)

        if len(one_gram2) >= 4:
            c_ba_four.append(len(token4) * 1.0 / len(set(four_gram2)))
        else:
            c_ba_four.append(0)

    df_['Jaccard_POS1gram'] = one
    df_['Jaccard_POS2gram'] = two
    df_['Jaccard_POS3gram'] = three
    df_['Jaccard_POS4gram'] = four

    df_['CAB_POS1gram'] = c_ab_one
    df_['CAB_POS2gram'] = c_ab_two
    df_['CAB_POS3gram'] = c_ab_three
    df_['CAB_POS4gram'] = c_ab_four

    df_['CBA_POS1gram'] = c_ba_one
    df_['CBA_POS2gram'] = c_ba_two
    df_['CBA_POS3gram'] = c_ba_three
    df_['CBA_POS4gram'] = c_ba_four

    return df_


def char_ngram(df_, sent):
    ctwo_gram = []  # Bigram tokenized list of list
    cthree_gram = []  # Trigram tokenized list of list
    cfour_gram = []  # Quadgram tokenized list of list
    cfive_gram = []  # Pentagram tokenized list of list

    for s in sent:
        ctwo_gram.append(nozip_ngram(s.lower(), n=2))
        cthree_gram.append(nozip_ngram(s.lower(), n=3))
        cfour_gram.append(nozip_ngram(s.lower(), n=4))
        cfive_gram.append(nozip_ngram(s.lower(), n=5))

    weight_two2 = []
    weight_three3 = []
    weight_four4 = []
    weight_five5 = [] 
    lcs1 = []

    jchar_substr = []
    cabchar_substr = []
    cbachar_substr = []

    for ind, row in df_.iterrows():

        match = SequenceMatcher(None, row['Sent1'], row['Sent2'])
        cmnsubstr = match.get_matching_blocks()

        jchar_substr.append(len(cmnsubstr)*1.0/len(set(row['Sent1']).union(row['Sent2'])))
        cabchar_substr.append(len(cmnsubstr)*1.0/len(set(row['Sent1'])))
        cbachar_substr.append(len(cmnsubstr)*1.0/len(set(row['Sent2'])))

        lcsubstr = match.find_longest_match(0, len(row['Sent1']), 0, len(row['Sent2']))
        lcs1.append(len(lcsubstr)*1.0/len(set(row['Sent1']).union(row['Sent2'])))

        # For 2-gram
        two_gram1 = nozip_ngram(row['Sent1'].lower(), n=2)
        two_gram2 = nozip_ngram(row['Sent2'].lower(), n=2)
        # For 3-gram
        three_gram1 = nozip_ngram(row['Sent1'].lower(), n=3)
        three_gram2 = nozip_ngram(row['Sent2'].lower(), n=3)
        # For 4-gram
        four_gram1 = nozip_ngram(row['Sent1'].lower(), n=4)
        four_gram2 = nozip_ngram(row['Sent2'].lower(), n=4)
        # For 5-gram
        five_gram1 = nozip_ngram(row['Sent1'].lower(), n=5)
        five_gram2 = nozip_ngram(row['Sent2'].lower(), n=5)

        token2 = set(two_gram1).intersection(two_gram2)
        token3 = set(three_gram1).intersection(three_gram2)
        token4 = set(four_gram1).intersection(four_gram2)
        token5 = set(five_gram1).intersection(five_gram2)

        weight_two2.append(tfidf(ctwo_gram, token2))
        weight_three3.append(tfidf(cthree_gram, token3))
        weight_four4.append(tfidf(cfour_gram, token4))
        weight_five5.append(tfidf(cfive_gram, token5))

    df_['Jaccard Char'] = jchar_substr
    df_['CAB Char'] = cabchar_substr
    df_['CBA Char'] = cbachar_substr

    df_['ProbIDF c2gram'] = weight_two2
    df_['ProbIDF c3gram'] = weight_three3
    df_['ProbIDF c4gram'] = weight_four4
    df_['ProbIDF c5gram'] = weight_five5

    df_['LCSubstr'] = lcs1
    
    return df_


def prob_idf(df_, sent):
    vocabulary = set()
    word_idf = collections.defaultdict(lambda: 0)
    word_tf = collections.defaultdict(lambda: 0)
    for s in sent:
        tkn = tokenize(s.decode('utf-8').encode('ascii', 'ignore').strip())
        idfw = set(tkn)
        vocabulary.update(idfw)
        for word in idfw:
            word_idf[word] += 1
        for word in tkn:
            word_tf[word] += 1

    for word in vocabulary:
        word_idf[word] = math.log(len(sent) / float(word_idf[word]))
        word_tf[word] = word_tf[word]*1.0/len(sent) 

    cos = []

    for ind, row in df_.iterrows():
        s1_tfidf = word_idf.fromkeys(word_idf, 0)
        s2_tfidf = word_idf.fromkeys(word_idf, 0)

        s1 = tokenize(row['Sent1'].decode('utf-8').encode('ascii', 'ignore').strip())
        s2 = tokenize(row['Sent2'].decode('utf-8').encode('ascii', 'ignore').strip())

        for w in s1:
            s1_tfidf[w] = word_tf[w]*word_idf[w]

        for w in s2:
            s2_tfidf[w] = word_tf[w]*word_idf[w]

        # print("ind="+str(ind))
        # print(spatial.distance.cosine(s1_tfidf.values(), s2_tfidf.values()))
        cos.append(1 - spatial.distance.cosine(s1_tfidf.values(), s2_tfidf.values()))

    df_['TFIDF_cosine'] = cos
    
    return df_, word_idf, word_tf
