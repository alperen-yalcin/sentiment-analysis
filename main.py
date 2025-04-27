from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import tensorflow_datasets as tfds
import re


train_data, test_data = tfds.load(
    'imdb_reviews',
    split=['train', 'test'],
    as_supervised = True
)

train_examples = []
train_labels = []
for text, label in tfds.as_numpy(train_data):
    train_examples.append(text.decode('utf-8'))
    train_labels.append(label)
train_df = pd.DataFrame({
    'review': train_examples,
    'label': train_labels
})

test_examples = []
test_labels = []
for text, label in tfds.as_numpy(test_data):
    test_examples.append(text.decode('utf-8'))
    test_labels.append(label)
test_df = pd.DataFrame({
    'review': test_examples,
    'label': test_labels
})

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

train_df['cleaned_review'] = train_df['review'].apply(clean_text)
test_df['cleaned_review'] = test_df['review'].apply(clean_text)

X_train_text = train_df['cleaned_review']
y_train_labels = train_df['label']

X_test_text = test_df['cleaned_review']
y_test_labels = test_df['label']

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
 
X_train_vec = vectorizer.fit_transform(X_train_text)
X_test_vec = vectorizer.transform(X_test_text)
