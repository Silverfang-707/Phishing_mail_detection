import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the training data from a CSV file
data = pd.read_csv('mail.csv')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2)

# Create a CountVectorizer to transform the text data into numerical features
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Create and train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions on the test set and calculate the accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Predict whether a given text input is from a phishing email or not
text_input = ""
text_input = vectorizer.transform([text_input])
prediction = clf.predict(text_input)
print(f'Prediction: {prediction}')
