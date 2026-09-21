# Decision Tree + Bagging
Difficulty: Hard

Implement a binary-classification decision tree (CART, Gini impurity, axis-aligned binary splits) plus a bagging ensemble, progressing through seven stages.

**Tree representation:** a node is a dict.
- Leaf node: `{"leaf": True, "value": <majority class, int>}`
- Internal node: `{"leaf": False, "feature": <int>, "threshold": <float>, "left": <node>, "right": <node>}`

A sample goes LEFT if `x[feature] <= threshold`, RIGHT otherwise.

Ties in majority class are broken by choosing the smaller label.

## Stage 1: Impurity

Implement `gini_impurity(y)` — Gini impurity: `1 - sum_c(p_c^2)`. Empty `y` returns 0.0.

## Stage 2: Splitting

Implement `split_dataset(X, y, feature, threshold)` — partition into left (`<=`) and right (`>`) subsets.

## Stage 3: Scoring a Split

Implement `weighted_impurity(y_left, y_right)` — sample-size-weighted average Gini of both sides.

## Stage 4: Finding the Best Split

Implement `best_split(X, y)` — search all features and midpoint thresholds, return `(feature, threshold, impurity)` or `None`.

## Stage 5: Building the Tree

Implement `majority_class(y)` and `build_tree(X, y, depth, max_depth, min_samples_split)`.

## Stage 6: Prediction

Implement `predict_one(x, node)` and `predict(X, node)`.

## Stage 7: Bagging Ensemble

Implement `bootstrap_sample(X, y, seed)`, `fit_bagged_trees(X, y, ...)`, and `predict_bagged(X, trees)`.

```python
def solve(input_data):
    # input_data = {"test": "test_name"}
    # Dispatches to internal test validators.
```
