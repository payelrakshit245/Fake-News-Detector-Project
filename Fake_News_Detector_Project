from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample training data (manually added)
texts = [
    "NASA finds water on Mars",
    "Apple launches new iPhone with AI",
    "Aliens have landed in Delhi",
    "Government bans all smartphones",
    "COVID-19 vaccine approved worldwide",
    "World ends tomorrow",
    "Elon Musk buys the Moon",
    "UN announces global peace treaty",
]
labels = ["REAL", "REAL", "FAKE", "FAKE", "REAL", "FAKE", "FAKE", "REAL"]

# Step 1: Preprocess
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Step 2: Train model
model = LogisticRegression()
model.fit(X, labels)

# Step 3: Take input and predict
headline = input("📰 Enter a news headline: ")
headline_vec = vectorizer.transform([headline])
prediction = model.predict(headline_vec)

print(" This news is:", "FAKE " if prediction[0] == "FAKE" else "REAL ")
