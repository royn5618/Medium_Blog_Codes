# Random Forest

max_depth = [None, 5, 8, 15, 25, 30]
min_samples_leaf = [1, 2, 10, 50, 100]
min_samples_split = [1, 2, 10, 50, 100]
n_estimators = [100, 300, 500, 800, 1200]
max_features = [None, 'sqrt', 'log2']

# Logistic Regression

max_iter = [100, 200, 500, 1000]
C = [0.1, 0.5, 1, 10, 50, 100]
penalty = ['elasticnet', 'l1', 'l2']
solver = ['saga', 'liblinear', 'lbfgs']
fit_intercept = [True, False]
