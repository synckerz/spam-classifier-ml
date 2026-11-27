import pandas as pd
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


# Load the SMS dataset
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

# Convert text labels into numbers
data["label"] = data["label"].map({"ham": 0, "spam": 1})
data = data.dropna(subset=["label", "message"])
data["label"] = data["label"].astype(int)

X = data["message"]
Y = data["label"]

# Split the dataset into training and testing data
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

# Create the text classification pipeline
spam_model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

# Train the model
spam_model.fit(X_train, Y_train)
print("\nModel Training Complete!")

# Predict labels for the test data
Y_pred = spam_model.predict(X_test)
print("\nPredictions Completed")
print("Total Predictions:", len(Y_pred))

# Evaluate model performance
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, zero_division=0)
recall = recall_score(Y_test, Y_pred, zero_division=0)

print("\n===== MODEL EVALUATION =====")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

# Display the confusion matrix
cm = confusion_matrix(Y_test, Y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

print("\n===== CONFUSION MATRIX =====")
print(cm)
print("\nTrue Negative (TN):", tn)
print("False Positive (FP):", fp)
print("False Negative (FN):", fn)
print("True Positive (TP):", tp)

# Display precision, recall and F1-score for each class
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    Y_test,
    Y_pred,
    target_names=["Not Spam", "Spam"],
    zero_division=0
))

# Test the trained model with new messages
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
