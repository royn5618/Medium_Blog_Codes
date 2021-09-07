import pandas as pd
from IPython.display import display

import time

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier


# initialize a dataframe to store model performance
df_performance_metrics = pd.DataFrame(columns=[
    'Model', 'Accuracy_Training_Set', 'Accuracy_Test_Set', 'Precision',
    'Recall', 'f1_score', 'Training Time (secs)'
])

# initialize a list to store the models
list_models_trained = []

# create a function that takes in the model and index of the model in the list list_init_models
def get_initial_performance_metrics(model, i):
    # model name
    model_name = type(model).__name__
    # time keeping
    start_time = time.time()
    print("Training {} model...".format(model_name))
    # Fitting of model
    model.fit(X_train, y_train)
    print("Completed {} model training.".format(model_name))
    elapsed_time = time.time() - start_time
    # Time Elapsed
    print("Time elapsed: {:.2f} s.".format(elapsed_time))
    # Predictions
    y_pred = model.predict(X_test)
    # Add to ith row of dataframe - metrics
    df_performance_metrics.loc[i] = [
        model_name,
        model.score(X_train, y_train),
        model.score(X_test, y_test),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred), 
        "{:.2f}".format(elapsed_time)
    ]
    # keep a track of trained models
    list_models_trained.append(model)
    print("Completed {} model's performance assessment.".format(model_name))
    
# initialize a list of models of interest   
list_init_models = [LogisticRegression(),
                    MultinomialNB(),
                    DecisionTreeClassifier(),
                    RandomForestClassifier(),
                    GradientBoostingClassifier(),
                    AdaBoostClassifier()]

# execute the function get_initial_performance_metrics for each ML model in the list
for n, model in enumerate(list_init_models):
    get_initial_performance_metrics(model, n)

display(df_performance_metrics)
