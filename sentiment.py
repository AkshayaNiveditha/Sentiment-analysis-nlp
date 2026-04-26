from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample data
texts = ["I love this product", "This is bad", "Amazing experience", "Worst service"]
labels = [1, 0, 1, 0]

# Convert text to numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = LogisticRegression()
model.fit(X, labels)

# Test
test = ["I hate this"]
test_vec = vectorizer.transform(test)

prediction = model.predict(test_vec)

print("Sentiment:", "Positive" if prediction[0] == 1 else "Negative")