# SMS Spam Classifier
# This file contains the reusable model creation and training functions.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def create_spam_model():
    """Create a text classification pipeline."""
    return Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB())
    ])


def train_spam_model(messages, labels):
    """Train and return a spam classification model."""
    model = create_spam_model()
    model.fit(messages, labels)
    return model
