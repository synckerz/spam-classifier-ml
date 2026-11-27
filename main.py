from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

import pandas as pd


# ==========================================
# 1. LOAD REAL DATASET
# ==========================================

data = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("\n===== DATASET INFORMATION =====")
print(data.head())
print("\nLabel Distribution:")
print(data["label"].value_counts())
print("\nDataset Shape:", data.shape)


# ==========================================
# 2. DATA PREPROCESSING
# ==========================================

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

# Remove missing or invalid records
data = data.dropna(subset=["label", "message"])

data["label"] = data["label"].astype(int)

X = data["message"]
Y = data["label"]


# ==========================================
# 3. SPLIT DATASET
# ==========================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

print("\n===== DATASET SPLIT =====")
print("Training Messages:", len(X_train))
print("Training Labels:", len(Y_train))
print("Testing Messages:", len(X_test))
print("Testing Labels:", len(Y_test))


# ==========================================
# 4. CREATE AI MODEL
# ==========================================

spam_model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])


# ==========================================
# 5. TRAIN AI MODEL
# ==========================================

spam_model.fit(X_train, Y_train)

print("\nModel Training Complete!")


# ==========================================
# 6. MAKE PREDICTIONS
# ==========================================

Y_pred = spam_model.predict(X_test)

print("\nPredictions Completed")
print("Total Predictions:", len(Y_pred))


# ==========================================
# 7. EVALUATE MODEL
# ==========================================

accuracy = accuracy_score(Y_test, Y_pred)

precision = precision_score(
    Y_test,
    Y_pred,
    zero_division=0
)

recall = recall_score(
    Y_test,
    Y_pred,
    zero_division=0
)

print("\n===== MODEL EVALUATION =====")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    Y_test,
    Y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("\n===== CONFUSION MATRIX =====")
print(cm)

print("\nTrue Negative (TN):", tn)
print("False Positive (FP):", fp)
print("False Negative (FN):", fn)
print("True Positive (TP):", tp)


# ==========================================
# 9. CLASSIFICATION REPORT
# ==========================================

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        Y_test,
        Y_pred,
        target_names=["Not Spam", "Spam"],
        zero_division=0
    )
)


# ==========================================
# 10. TEST NEW MESSAGES
# ==========================================

new_messages = [
    "Congratulations! You have won a free prize",
    "Hello, I will see you at the meeting tomorrow",
    "Claim your free cash reward now"
]

new_predictions = spam_model.predict(new_messages)

print("\n===== NEW MESSAGE TESTING =====")

for message, prediction in zip(new_messages, new_predictions):
    result = "Spam" if prediction == 1 else "Not Spam"
    print(f"{message} ---> {result}")


print("\n===== PROGRAM COMPLETED SUCCESSFULLY =====")
