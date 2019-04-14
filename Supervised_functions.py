import time

import pandas as pd
import scipy.stats as meas
import sklearn
import spacy
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

import lexical_syntactic as lex_s
import semantic_sim as sem

LATIN_1_CHARS = (            # To convert unicodes into understandable punctuations
    ('\xe2\x80\x99', "'"),
    ('\xef\xac\x81', ''),
    ('- ', ''),
    (' - ', ''),
    ('\xc3\xa9', 'e'),
    ('\xe2\x80\x90', '-'),
    ('\xe2\x80\x91', '-'),
    ('\xe2\x80\x92', '-'),
    ('\xe2\x80\x93', '-'),
    ('\xe2\x80\x94', '-'),
    ('\xe2\x80\x94', '-'),
    ('\xe2\x80\x98', "'"),
    ('\xe2\x80\x9b', "'"),
    ('\xe2\x80\x9c', '"'),
    ('\xe2\x80\x9c', '"'),
    ('\xe2\x80\x9d', '"'),
    ('\xe2\x80\x9e', '"'),
    ('\xe2\x80\x9f', '"'),
    ('\xe2\x80\xa6', '...'),
    ('\xe2\x80\xb2', "'"),
    ('\xe2\x80\xb3', "'"),
    ('\xe2\x80\xb4', "'"),
    ('\xe2\x80\xb5', "'"),
    ('\xe2\x80\xb6', "'"),
    ('\xe2\x80\xb7', "'"),
    ('\xe2\x81\xba', "+"),
    ('\xe2\x81\xbb', "-"),
    ('\xe2\x81\xbc', "="),
    ('\xe2\x81\xbd', "("),
    ('\xe2\x81\xbe', ")"),
    ('\xef\x82\xb7', ''), 
    ('\xef\x82\xa7', '')
)


def load_data(test_file=None):
    print "Hell"
    train_len = 0
    df_ = pd.read_csv("Semval.csv")
    if test_file is not None:
        train_len = len(df_)	
        test = pd.read_csv("Checker.csv")
        print "India" 
        if 'Similarity Score' in test.columns:
            test = test.drop(['Similarity Score'], axis=1)
        if 'SemEval' not in test.columns:
            test['SemEval'] = [0]*len(test)		
        df_ = df_.append(test)
    
    for ind, row in df_.iterrows():
        a = row['Sent1']
        for _hex, _char in LATIN_1_CHARS:
            a = a.replace(_hex, _char)
        a = a.decode('utf-8').encode('ascii', 'ignore').strip()
        df_.loc[ind, 'Sent1'] = a

        b = row['Sent2']
        for _hex, _char in LATIN_1_CHARS:
            b = b.replace(_hex, _char)
            # print row['Sent2']
        b = b.decode('utf-8').encode('ascii', 'ignore').strip()
        df_.loc[ind, 'Sent2'] = b

    sent = []
    for ind, row in df_.iterrows():
        if row['Sent1'] not in sent:  # Getting all unique text
            a = row['Sent1']
            for _hex, _char in LATIN_1_CHARS:
                a = a.replace(_hex, _char)
            sent.append(a.decode('utf-8').encode('ascii', 'ignore').strip())
        if row['Sent2'] not in sent:  # Getting all unique text
            b = row['Sent2']
            for _hex, _char in LATIN_1_CHARS:
                b = b.replace(_hex, _char)
            sent.append(b.decode('utf-8').encode('ascii', 'ignore').strip())

    return df_, sent, train_len


def evaluation(predict, gold):
    """
    pearsonr of predict and gold
    Args:
        predict: list
        gold: list
    Returns:
        pearson of predict and gold
    """
    pearsonr = meas.pearsonr(predict, gold)[0]
    return pearsonr


def split_data(a, train_len):
    print("Splitting data in 80:20..")
    if train_len == 0:
        train, test = sklearn.model_selection.train_test_split(a, train_size=0.80, test_size=0.20,
                                                               random_state=33, stratify=a['SemEval'])
    else:
        train = a.iloc[0:train_len, :]
        test = a.iloc[train_len:len(a), :]
        # print test
    print("Splitting Done..")
    
    return train, test


def dataset_for_modelling(train, test):
    y_train = train.loc[:, 'SemEval']
    y_test = test.loc[:, 'SemEval']
    x_train = train.drop(['Sent1', 'Sent2', 'SemEval'], axis=1)
    x_test = test.drop(['Sent1', 'Sent2', 'SemEval'], axis=1)

    return x_train, y_train, x_test, y_test


def _svm(x_train, y_train, x_test, y_test):
    
    print("SVM starts...")
    clf = svm.SVC(kernel='rbf', C=15000, gamma=0.009)
    clf.fit(x_train, y_train)
    p = clf.predict(x_test)
    # print p
    acc = round(accuracy_score(y_test, p) * 100, 2)
    # acc1 = 0
    acc1 = evaluation(y_test, p)*100
    print("SVM Ends...")
    
    return acc1, acc, p


def nn(x_train, y_train, x_test, y_test):
    print("NN starts...")
    clf = MLPClassifier(solver='adam', activation='relu', alpha=0.01, hidden_layer_sizes=(11, 29, 28, 10),
                        random_state=2)
    clf.fit(x_train, y_train)

    p = clf.predict(x_test)
    # print p
    acc = round(accuracy_score(y_test, p) * 100, 2)
    # acc1 = 0
    acc1 = evaluation(y_test, p) * 100
    print("NN Ends...")

    return acc1, acc, p


def sentence_sim():
    start_time = time.time()
    test_file = 'Checker.csv'
    lang = 'en'
    #test_file = None

    if lang == 'en':  # 802MB
        nlp = spacy.load('en_core_web_lg')
    elif lang == 'es':  # 98MB
        nlp = spacy.load('es_core_news_md')
    elif lang == 'fr':  # 112MB
        nlp = spacy.load('fr_core_news_md')

    print("Start")
    begin = time.time()
    df_, sent, train_len = load_data(test_file=test_file)
    print "Time:" + str(time.time() - begin)
    # df_ = df_.loc[0:115, :]
    print

    print("Part1")
    begin = time.time()
    df_ = lex_s.lemma_ngram(df_, sent)
    print "Time:" + str(time.time() - begin)
    print

    print("Part4")
    begin = time.time()
    df_, word_idf, word_tf = lex_s.prob_idf(df_, sent)
    print "Time:" + str(time.time() - begin)
    print

    print("Part5")
    begin = time.time()
    df_ = sem.semantic_composition(df_, word_idf, word_tf, nlp)
    print "Time:" + str(time.time() - begin)
    print

    print("Part6")
    begin = time.time()
    df_ = sem.para2vec(df_, nlp)
    print "Time:" + str(time.time() - begin)
    print

    # df_.to_csv("Hed2.csv", index=False)

    print("Part2")
    begin = time.times()
    df_ = lex_s.pos_ngram(df_, nlp)
    print "Time:" + str(time.time() - begin)
    print

    """
    print("Part3")
    begin = time.time()
    df_ = lex_s.char_ngram(df_, sent)
    print "Time:" + str(time.time() - begin)
    print
    """

    # df_.to_csv("Hed1.csv", index=False)

    train, test = split_data(df_, train_len)
    x_train, y_train, x_test, y_test = dataset_for_modelling(train, test)

    acc5, acc_s, pred = _svm(x_train, y_train, x_test, y_test)

    if test_file is not None:
        dd = pd.read_csv(test_file)
        dd['Similarity Score'] = pred
        if 'SemEval' in dd.columns:
            dd = dd.drop(['SemEval'], axis=1)
        dd.to_csv(test_file, index=False)

    print("Pearson Correlation:" + str(acc5))
    print("Accuracy Score:" + str(acc_s))
    print("Time taken by System: %s seconds" % (time.time() - start_time))
    # print(df_.head(n=6))
    print("NN")
    acc5, acc_s, pred = nn(x_train, y_train, x_test, y_test)
    print("Pearson Correlation:" + str(acc5))
    print("Accuracy Score:" + str(acc_s))
    print("Time taken by System: %s seconds" % (time.time() - start_time))


def main():
    sentence_sim()


if __name__ == '__main__':
    main()
