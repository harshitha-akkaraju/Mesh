# Mesh

Mesh is a Python script to classify whether or not a certain text message (message) is spam. I implemented a Naive Bayes
Classifier, an ML technique, which involves applying the Bayes theorem assuming that the occurance of certain words and
 a message being spam are independent.

## Dataset
* [SMS dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset): Over 5000 messages that were labeled as spam or
ham
* Size of the training data set: 3900 messages
* Size of the evaluation data set: 1672 messages

## Results
With minor changes to the Naive Bayes algorithm such as Laplace smoothing and transforming the computations into Log space
(to avoid floating point multiplication), the model was able to achieve the following results.

```
Correctly classified: 96.23 % messages
False negative stats: 0.54 %
False positive stats: 3.23 %
```

Read my notes on the Naive Bayes classifier and the run time of this algorithm [here]().