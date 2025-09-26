import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

# ✅ Set UTF-8 encoding to avoid Unicode errors
sys.stdout.reconfigure(encoding='utf-8')

# Load dataset (Replace with your dataset path)
file_path = r'c:/finalyear/backend/fakedetection/Dataset/news.csv'

try:
    df = pd.read_csv(file_path)  # Ensure file exists
except FileNotFoundError:
    print(f"❌ Error: Dataset not found at {file_path}")
    exit()

# Preprocessing
if 'text' not in df.columns or 'label' not in df.columns:
    print("❌ Error: Missing 'text' or 'label' column in the dataset!")
    exit()

X = df['text']
y = df['label'].map({'FAKE': 1, 'REAL': 0})  # Convert labels to binary (1 = Fake, 0 = Real)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ✅ Save TF-IDF vectorizer
pkl_path = r"c:/finalyear/backend/fakedetection/tfidf_vectorizer.pkl"

try:
    with open(pkl_path, "wb") as f:
        pickle.dump(tfidf, f)
    print("TF-IDF vectorizer saved successfully!")
except Exception as e:
    print(f"❌ Error saving TF-IDF vectorizer: {e}")

# ✅ Load TF-IDF vectorizer
try:
    with open(pkl_path, "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    print("TF-IDF model loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: TF-IDF file not found at {pkl_path}")
    exit()

# Initialize models
models = {
    "passive_aggressive": PassiveAggressiveClassifier(max_iter=50),
    "logistic_regression": LogisticRegression(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "naive_bayes": MultinomialNB(),
    "decision_tree": DecisionTreeClassifier()
}

# Train & Save Models
model_path = r'c:/finalyear/backend/fakedetection/models/'

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.2f}")

    # Save the trained model
    model_file = f"{model_path}{name}_model.pkl"
    try:
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Model '{name}' saved successfully!")
    except Exception as e:
        print(f"❌ Error saving model '{name}': {e}")

print("✅ All models trained and saved!")
