# Naive Bayes Classifier
Difficulty: Medium

Implement a Gaussian Naive Bayes classifier from scratch using NumPy, progressing through six stages.

Gaussian Naive Bayes models each feature as a Gaussian distribution per class, then uses Bayes' theorem to predict:

$$P(y|x) \propto P(y) \prod_j P(x_j | y)$$

## Stage 1: Class Priors

Implement `compute_priors(y, n_classes)` — return the prior probability of each class (fraction of training samples).

## Stage 2: Class Statistics

Implement `compute_class_stats(X, y, n_classes)` — return per-class means and variances for each feature.

## Stage 3: Log-Likelihood

Implement `gaussian_log_likelihood(x, mean, var)` — compute the log of the Gaussian PDF. Add a small epsilon to variance for numerical stability.

## Stage 4: Log-Posterior

Implement `compute_log_posteriors(X, priors, means, variances)` — compute unnormalized log-posterior for each class.

## Stage 5: Prediction

Implement `fit(X, y, n_classes)` — return a model dict with priors, means, variances.

Implement `predict(X, model)` — predict class labels by taking the argmax of log-posteriors.

## Stage 6: Evaluation

Implement `accuracy(y_pred, y_true)` — fraction correct.

Implement `fit_and_evaluate(X_train, y_train, X_test, y_test, n_classes)` — fit and return test accuracy.

```python
def solve(input_data):
    # input_data = {"test": "test_name"}
    # Dispatches to internal test validators.
```
