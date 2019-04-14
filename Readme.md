# Sentence Similarity

In this project, we are trying to find out how much the two sentences are similar on the
scale of [0,5]. [0,5] is the standard evaluation used by Semantic Textual Similarity (STS). It
stimulate research in this area and encourage the development of creative new
approaches to modeling sentence level semantics, the STS shared task has been held
annually since 2012, as part of the SemEval/SEM family of workshops. The selection of
STS datasets include text from image captions, news headlines and user forums.

We are
extracting various feature that could help us further in prediction the similarity value.
We have tried to replicate the result as mentioned in the paper “UWB at SemEval-2016
Task 1: Semantic Textual Similarity using Lexical, Syntactic, and Semantic Information”
using unsupervised learning.
The various features :
1. Lemma n-gram overlaps: Here we are converting the sentence into word-level n-gram
vector and are further using Jaccard Similarity and Containment coefficient. Further we are
finding the Longest Common Subsequence. Along with it we are also finding out the result
of probabilities times idf value for all n-gram. Here n є {1, 2, 3, 4}.
2. POS n-gram overlaps: Here we first converted the sentence into part-of-speech(POS) tag
and then into word-level n-gram vector and are further using Jaccard Similarity and
Containment coefficient. Here n є {1, 2, 3, 4}.
3. Character n-gram overlaps: Here we are converting the sentence into character level n-
gram vector and are further using Jaccard Similarity and Containment coefficient. Further
we are find the Longest Common Substring. Along with it we are also finding out the result
of probabilities times idf value for all n-gram. Here n є {2, 3, 4, 5}.
4. TF-IDF: For each word in the sentence, we compute the TF-IDF scores. Given vocabulary |
V|, we get the vector of length |V| for each sentence. Now we compute the similarity scores
using cosine similarity function.
5. Semantic composition: In this we used Wikipedia pre-trained model and for each token of
the sentence we used word2vec to find the vector of that particular token, further multiply it
with IDF score and summing the vector so as to get a single scaler value. So a vector of
vocabulary |V| is created. Now we compute the similarity scores using cosine similarity
function.
6. Paragraph2Vec: In this we used Doc2Vec of gensim package to find out the vector of the
sentence given it is a TaggedDocument and further applied cosine similarity.
The model was trained with several algorithm which include Linear regression, Gaussian process
Classifier, SVM regression, XGBoost and MLP Classifier. After parameter tuning for each
algorithm, it was observed that SVM surpasses all other algorithms (we took Pearson Correlation
as evaluation mechanism). So finally model is trained with SVM.