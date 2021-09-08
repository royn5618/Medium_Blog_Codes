# A function to get the desired metrics while working with multiple model training procedures
def print_classification_metrics(y_train, train_pred, y_test, test_pred, return_performance=True):
    dict_performance = {'Training Accuracy: ': accuracy_score(y_train, train_pred),
                        'Training f1-score: ': f1_score(y_train, train_pred),
                        'Accuracy: ': accuracy_score(y_test, test_pred),
                        'Precision: ': precision_score(y_test, test_pred),
                        'Recall: ': recall_score(y_test, test_pred),
                        'f1-score: ': f1_score(y_test, test_pred)}
    for key, value in dict_performance.items():
        print("{} : {}".format(key, value))
    if return_performance:
        return dict_performance
    
