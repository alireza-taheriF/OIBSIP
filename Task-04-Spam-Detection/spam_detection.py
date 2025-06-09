# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from collections import Counter
import re

# Load dataset
data_path = 'Data/spam.csv'
df = pd.read_csv(data_path, encoding='latin-1')

# Display few first rows
print('First few rows of the dataset')
print(df.head())

# Rename columns and drop unnecessary columns
df = df.rename(columns={'v1': 'label', 'v2': 'text'})
df = df[['label', 'text']]  # Keep only label and text columns

# Convert labels to binary (0 for ham, 1 for spam)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Check basic info
print('Dataset Info: ')
print(df.info())
print('\nMissing Values: ')
print(df.isnull().sum())

# Split features and target 
X = df['text']
y = df['label']

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print('\nNaive Bayes Classification Report: ')
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Confusion Matrix for Naive Bayes
cm_nb = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix - Naive Bayes')
plt.savefig('figures/confusion_matrix_nb.png')
plt.show()

# Word Clouds
# Spam word cloud
spam_text = ' '.join(df[df['label'] == 1]['text'])
wordcloud_spam = WordCloud(width=800, height=400, background_color='white').generate(spam_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_spam, interpolation='bilinear')
plt.title('Word Cloud for Spam Emails')
plt.axis('off')
plt.savefig('figures/wordcloud_spam.png')
plt.show()

# Ham word cloud
ham_text = ' '.join(df[df['label'] == 0]['text'])
wordcloud_ham = WordCloud(width=800, height=400, background_color='white').generate(ham_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_ham, interpolation='bilinear')
plt.title('Word Cloud for Ham Emails')
plt.axis('off')
plt.savefig('figures/wordcloud_ham.png')
plt.show()

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test)

# Evaluate
print('\nRandom Forest Classification Report: ')
print(classification_report(y_test, y_pred_rf, target_names=['Ham', 'Spam']))

# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix - Random Forest')
plt.savefig('figures/confusion_matrix_rf.png')
plt.show()

# Optimization with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print('\nBest Parameters: ', grid_search.best_params_)
best_rf_model = grid_search.best_estimator_
y_pred_best = best_rf_model.predict(X_test)
print('\nBest Random Forest Classification Report: ')
print(classification_report(y_test, y_pred_best, target_names=['Ham', 'Spam']))

# Number of common words: Checking the most common words in spam and ham
spam_words = ' '.join(df[df['label'] == 1]['text']).lower().split()
ham_words = ' '.join(df[df['label'] == 0]['text']).lower().split()
print('\nTop 10 Most Common Words in Spam: ', Counter(spam_words).most_common(10))
print('Top 10 Most Common Words in Ham: ', Counter(ham_words).most_common(10))

# Compare models
results = pd.DataFrame({
    'Model': ['Naive Bayes', 'Random Forest', 'Best Random Forest'],
    'Accuracy': [
        classification_report(y_test, y_pred, output_dict=True)['accuracy'],
        classification_report(y_test, y_pred_rf, output_dict=True)['accuracy'],
        classification_report(y_test, y_pred_best, output_dict=True)['accuracy']
    ]
})
print("\nModel Comparison:")
print(results)

# More advanced preprocessing
def preprocess_text(text):
    text = re.sub(r'[[^\w\s]', '', text.lower())
    return text
df['text'] = df['text'].apply(preprocess_text)