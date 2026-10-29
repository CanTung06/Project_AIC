"""One trained estimator with its validation-selected alert threshold."""

import numpy as np


class ThresholdedClassifier:
    def __init__(self, estimator, threshold, model_name):
        self.estimator = estimator
        self.threshold = float(threshold)
        self.model_name = str(model_name)

    @property
    def named_steps(self):
        return self.estimator.named_steps

    @property
    def classes_(self):
        return self.estimator.classes_

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        scores = self.predict_proba(X)[:, 1]
        return (scores >= self.threshold).astype(np.int8)
