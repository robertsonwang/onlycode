# Boosting with Decision Stumps (AdaBoost)
Difficulty: Hard

Implement binary boosting with decision stumps, progressing through seven stages.

Use labels in {-1, +1}.

A stump is represented as a dict:
- `{"feature": int, "threshold": float, "polarity": int}`

Prediction rule for one stump:
- If `polarity == 1`: predict `+1` when `x[feature] <= threshold`, else `-1`
- If `polarity == -1`: flip the sign

## Stage 1: Weighted Error

Implement `weighted_error(y_true, y_pred, w)`:
- return `sum(w_i for i where y_pred_i != y_true_i)`

## Stage 2: Stump Prediction

Implement `stump_predict(X, feature, threshold, polarity)` returning a vector in {-1, +1}.

## Stage 3: Best Stump Search

Implement `best_stump(X, y, w)`:
- search every feature and midpoint threshold between sorted unique values
- test both polarities
- return `(stump_dict, error)` with the minimum weighted error

## Stage 4: Alpha Weight

Implement `compute_alpha(err, eps=1e-12)`:
- clip err into `[eps, 1 - eps]`
- return `0.5 * log((1 - err) / err)`

## Stage 5: Weight Update

Implement `update_weights(w, y, y_pred, alpha)`:
- `w_i <- w_i * exp(-alpha * y_i * y_pred_i)`
- normalize so weights sum to 1

## Stage 6: Training Loop

Implement `fit_adaboost(X, y, n_estimators=20)`:
- initialize uniform weights
- for each round: find stump, compute alpha, update weights
- return a model dict with stumps and alphas

## Stage 7: Inference and Evaluation

Implement:
- `predict_scores(X, model)`  -> real-valued margins
- `predict_adaboost(X, model)` -> sign of margin in {-1, +1}
- `accuracy(y_true, y_pred)`   -> fraction correct

```python
def solve(input_data):
    # input_data = {"test": "test_name"}
    # Dispatches to internal test validators.
```
