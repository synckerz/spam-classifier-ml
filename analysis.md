# SMS Spam Classification – Analysis

The aim of this project was to develop a simple AI model that can classify SMS messages as spam or not spam. I used Python and the Scikit-learn library to implement a Multinomial Naive Bayes classifier. The model was trained using the SMS Spam Collection dataset, which contains 5,572 messages.

The dataset was divided into 80% training data and 20% testing data. CountVectorizer was used to convert text messages into numerical features before training the model.

The model achieved an accuracy of 98.74%, precision of 98.56% and recall of 91.95%. These results show that the model performed well when classifying messages in the test dataset. The confusion matrix showed 964 correctly identified legitimate messages and 137 correctly identified spam messages. However, 12 spam messages were incorrectly classified as legitimate, while 2 legitimate messages were classified as spam. This shows that the model can still make classification errors.

There are also ethical considerations when using this type of AI model. The dataset contains personal communication examples, so privacy and responsible data use are important. Although the dataset is publicly available, this does not mean that all message data should be used without considering privacy.

Another limitation is that the dataset may not represent all users, languages or communication styles. As a result, the model may perform differently when it receives messages from different groups or in different languages. The model may also struggle with new spam techniques that were not included in its training data.

Overall, the project demonstrated how machine learning can be used for SMS spam detection. Although the model achieved good results, its predictions should not be treated as completely reliable.
