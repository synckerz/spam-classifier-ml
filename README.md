# SMS Spam Classifier

A simple SMS spam classification project developed with Python and Scikit-learn.

## Project overview

This project uses a Multinomial Naive Bayes model to classify SMS messages as **Spam** or **Not Spam**. It uses the SMS Spam Collection dataset and evaluates the model on a separate test set.

## Technologies

- Python
- Pandas
- Scikit-learn
- CountVectorizer
- Multinomial Naive Bayes

## Dataset

The project uses the SMS Spam Collection dataset. The loaded dataset contains 5,572 messages:

- 4,825 legitimate messages (ham)
- 747 spam messages

The labels are converted to numeric values: ham = 0 and spam = 1.

## Model workflow

1. Load and inspect the dataset.
2. Convert the message labels into numeric values and remove missing rows.
3. Split the data into 80% training and 20% testing sets, using stratified sampling.
4. Convert message text into word-count features with CountVectorizer.
5. Train a Multinomial Naive Bayes classifier.
6. Evaluate predictions using accuracy, precision, recall, a confusion matrix and a classification report.
7. Test the trained model with new example messages.

## Evaluation results

Results from the recorded run:

| Metric | Result |
|---|---:|
| Accuracy | 98.74% |
| Precision | 98.56% |
| Recall | 91.95% |

Confusion matrix:

| | Predicted Not Spam | Predicted Spam |
|---|---:|---:|
| Actual Not Spam | 964 | 2 |
| Actual Spam | 12 | 137 |

The model correctly classified 1,101 of the 1,115 test messages. It missed 12 spam messages and incorrectly marked 2 legitimate messages as spam.

## Run the project

1. Clone or download this repository.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure the dataset file is located at `dataset/SMSSpamCollection`.
4. Run:

   ```bash
   python main.py
   ```

## Project structure

```text
spam-classifier-ml/
├── dataset/
│   └── SMSSpamCollection
├── results/
│   └── evaluation_results.txt
├── main.py
├── spam_classifier.py
├── README.md
├── requirements.txt
└── analysis.md
```

## Limitations and ethical considerations

The model does not detect every spam message. Its performance may change with different languages, communication styles or newer spam techniques. The dataset contains message text, so privacy and responsible data handling should be considered. A public dataset should still be used carefully.

## Dataset source

UCI Machine Learning Repository – SMS Spam Collection:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection
