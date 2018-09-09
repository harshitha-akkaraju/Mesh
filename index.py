import pandas as pd
import math

def mesh():
    # set up
    file_path = "./spam.csv"
    spam_data = pd.read_csv(file_path, ",", engine='python')
    spam_data = spam_data[["v1", "v2"]]  # ignore empty columns
    spam_data.columns = ["label", "sms_content"]  # rename columns

    # extract training and evaluation data (70:30 ratio)
    training_data = spam_data[1:3901]
    evaluation_data = spam_data[3901: 5573]

    spam_count = len(training_data[training_data.label == 'spam'])
    non_spam_count = len(training_data[training_data.label == 'ham'])
    all_sms = len(training_data)

    # essential variables
    prob_spam = spam_count / float(all_sms)
    prob_not_spam = non_spam_count / float(all_sms)
    positive = {}
    negative = {}

    for row in training_data.itertuples():
        process_sms_bow(row.label, row.sms_content, positive, negative)

    bow_model_labels = []
    for row in evaluation_data.itertuples():
        is_spam = bag_of_words_classifier(prob_spam, prob_not_spam, positive, negative, row.sms_content)
        if is_spam:
            bow_model_labels.append("spam")
        else:
            bow_model_labels.append("ham")

    bow_model_certainty = eval_predictions(bow_model_labels, evaluation_data.label)
    print(bow_model_certainty)


# function to obtain the word frequencies in an email
def process_sms_bow(label, content, positive, negative):
    words = content.split()
    for word in words:
        if label == "spam":
            if word in positive:
                positive[word] = positive[word] + 1
            else:
                positive[word] = 1
        elif label == "ham":
            if word in negative:
                negative[word] = negative[word] + 1
            else:
                negative[word] = 1


# computes a weight metric for the word depending on whether or not it is spam
def eval_word_bow(is_spam, word, positive, negative):
    if is_spam and word in positive:
        return positive[word] / float(len(positive))
    elif (not is_spam) and (word in negative):
        return negative[word] / float(len(negative))
    return 0.00001    # if it is an unseen word


# computes the probability that sms is spam
def eval_sms_bow(is_spam, content, positive, negative):
    result = 0
    words = content.split()
    for word in words:
        result += math.log(eval_word_bow(is_spam, word, positive, negative))
    return result


def bag_of_words_classifier(prob_spam, prob_not_spam, positive, negative, sms_content):
    is_spam = math.log(prob_spam) + eval_sms_bow(True, sms_content, positive, negative)
    is_not_spam = math.log(prob_not_spam) + eval_sms_bow(False, sms_content, positive, negative)
    return is_spam > is_not_spam


def eval_predictions(predictions, labeled):
    labeled = labeled.rename(lambda x: x - 3901)  # switch to 0 based indexing
    num_correct = 0
    for row in labeled.iteritems():
        index = row[0]
        label = row[1]
        if label == predictions[index]:
            num_correct = num_correct + 1
    return (num_correct / float(len(labeled))) * 100



if __name__ == '__main__':
  mesh()
