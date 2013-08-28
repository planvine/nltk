# Natural Language Toolkit: Interface to scikit-learn classifiers
#
# Author: Lars Buitinck <L.J.Buitinck@uva.nl>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT
"""
scikit-learn (http://scikit-learn.org) is a machine learning library for
Python, supporting most of the basic classification algorithms, including SVMs,
Naive Bayes, logistic regression and decision trees.

This package implement a wrapper around scikit-learn classifiers. To use this
wrapper, construct a scikit-learn classifier, then use that to construct a
SklearnClassifier. E.g., to wrap a linear SVM classifier with default settings,
do

>>> from sklearn.svm.sparse import LinearSVC
>>> from nltk.classify.scikitlearn import SklearnClassifier
>>> classif = SklearnClassifier(LinearSVC())

The scikit-learn classifier may be arbitrarily complex. E.g., the following
constructs and wraps a Naive Bayes estimator with tf-idf weighting and
chi-square feature selection:

>>> from sklearn.feature_extraction.text import TfidfTransformer
>>> from sklearn.feature_selection import SelectKBest, chi2
>>> from sklearn.naive_bayes import MultinomialNB
>>> from sklearn.pipeline import Pipeline
>>> pipeline = Pipeline([('tfidf', TfidfTransformer()),
...                      ('chi2', SelectKBest(chi2, k=1000)),
...                      ('nb', MultinomialNB())])
>>> classif = SklearnClassifier(pipeline)

(Such a classifier could be trained on word counts for text classification.)

The version of this module in NLTK 2.0.4 does not support lazy feature
extraction.

Lazy feature extraction allows memory usage to be reduced by storing features
only when they're used, meaning the entire feature set does not need to be
stored in memory. See http://nltk.org/api/nltk.classify.html#nltk.classify.util.apply_features
for more information.

In this version, SklearnClassifier.train has been modified to use the iterator
protocol, rather than storing all extracted features in memory.
"""

from nltk.classify.api import ClassifierI
from nltk.probability import DictionaryProbDist
from scipy.sparse import coo_matrix
from itertools import imap
from operator import itemgetter

try:
    import numpy as np
except ImportError:
    pass

class SklearnClassifier(ClassifierI):
    """Wrapper for scikit-learn classifiers."""

    def __init__(self, estimator, dtype=float, sparse=True):
        """
        :param estimator: scikit-learn classifier object.

        :param dtype: data type used when building feature array.
            scikit-learn estimators work exclusively on numeric data; use bool
            when all features are binary.

        :param sparse: Whether to use sparse matrices. The estimator must
            support these; not all scikit-learn classifiers do. The default
            value is True, since most NLP problems involve sparse feature sets.
        :type sparse: boolean.
        """
        self._clf = estimator
        self._dtype = dtype
        self._sparse = sparse

    def __repr__(self):
        return "<SklearnClassifier(%r)>" % self._clf

    def batch_classify(self, featuresets):
        X = self._convert(featuresets)
        y = self._clf.predict(X)
        return [self._index_label[int(yi)] for yi in y]

    def batch_prob_classify(self, featuresets):
        X = self._convert(featuresets)
        y_proba = self._clf.predict_proba(X)
        return [self._make_probdist(y_proba[i]) for i in xrange(len(y_proba))]

    def labels(self):
        return self._label_index.keys()

    def train(self, labeled_featuresets):
        """
        Train (fit) the scikit-learn estimator.

        Respect lazy feature extraction.

        >>> toks = [("token 1", True), ("token 2", False)]
        >>> labeled_featuresets = nltk.classify.util.apply_features(feature_func, toks, labeled=True)
        >>> SklearnClassifier.train(labeled_featuresets)

        In the above example the results of `feature_func("token 1")` and
        `feature_func("token 2")` need not both be stored in memory.

        :param labeled_featuresets: A list of classified featuresets,
            i.e., a list of tuples ``(featureset, label)``.

        :type labeled_featuresets: iterator
        """

        self._feature_index = {}
        self._index_label = []
        self._label_index = {}

        # Keep track of iterations so don't need to calculate length
        # again later.
        self._labeled_featuresets_len = 0
        for fs, label in labeled_featuresets:
            for f in fs.iterkeys():
                if f not in self._feature_index:
                    self._feature_index[f] = len(self._feature_index)
            if label not in self._label_index:
                self._index_label.append(label)
                self._label_index[label] = len(self._label_index)
            self._labeled_featuresets_len += 1

        # zip greedily consumes labeled_featuresets, all feature sets are stored
        # in `featuresets` and passed to convert
        # featuresets, labels = zip(*labeled_featuresets)
        featuresets = imap(itemgetter(0), labeled_featuresets)
        labels = imap(itemgetter(1), labeled_featuresets)

        X = self._convert(featuresets)
        y = np.array([self._label_index[l] for l in labels])

        self._clf.fit(X, y)

        return self

    def _convert(self, featuresets):
        if self._sparse:
            return self._featuresets_to_coo(featuresets)
        else:
            return self._featuresets_to_array(featuresets)

    def _featuresets_to_coo(self, featuresets):
        """Convert featuresets to sparse matrix (COO format)."""

        i_ind = []
        j_ind = []
        values = []

        for i, fs in enumerate(featuresets):
            for f, v in fs.iteritems():
                try:
                    j = self._feature_index[f]
                    i_ind.append(i)
                    j_ind.append(j)
                    values.append(self._dtype(v))
                except KeyError:
                    pass

        shape = (i + 1, len(self._feature_index))
        return coo_matrix((values, (i_ind, j_ind)), shape=shape, dtype=self._dtype)

    def _featuresets_to_array(self, featuresets):
        """Convert featureset to Numpy array."""

        X = np.zeros((self._labeled_featuresets_len, len(self._feature_index)),
            dtype=self._dtype)

        for i, fs in enumerate(featuresets):
            for f, v in fs.iteritems():
                try:
                    X[i, self._feature_index[f]] = self._dtype(v)
                except KeyError:    # feature not seen in training
                    pass

        return X

    def _make_probdist(self, y_proba):
        return DictionaryProbDist(dict((self._index_label[i], p)
                                       for i, p in enumerate(y_proba)))


if __name__ == "__main__":
    from nltk.classify.util import names_demo, binary_names_demo_features
    try:
        from sklearn.linear_model.sparse import LogisticRegression
    except ImportError:     # separate sparse LR to be removed in 0.12
        from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import BernoulliNB

    print("scikit-learn Naive Bayes:")
    names_demo(SklearnClassifier(BernoulliNB(binarize=False), dtype=bool).train,
               features=binary_names_demo_features)
    print("scikit-learn logistic regression:")
    names_demo(SklearnClassifier(LogisticRegression(), dtype=np.float64).train,
               features=binary_names_demo_features)
