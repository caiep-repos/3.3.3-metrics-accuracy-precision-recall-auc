import unittest
import numpy as np
from sklearn.metrics import accuracy_score as sklearn_accuracy, precision_score as sklearn_precision, recall_score as sklearn_recall, f1_score as sklearn_f1, roc_auc_score as sklearn_roc_auc
from assignment  import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class TestClassificationMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        self.y_pred = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 0])
        self.y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.52, 0.51, 0.50])

    def test_accuracy(self):
        self.assertAlmostEqual(accuracy_score(self.y_true, self.y_pred), sklearn_accuracy(self.y_true, self.y_pred))

    def test_precision(self):
        self.assertAlmostEqual(precision_score(self.y_true, self.y_pred), sklearn_precision(self.y_true, self.y_pred))

    def test_recall(self):
        self.assertAlmostEqual(recall_score(self.y_true, self.y_pred), sklearn_recall(self.y_true, self.y_pred))

    def test_f1_score(self):
        self.assertAlmostEqual(f1_score(self.y_true, self.y_pred), sklearn_f1(self.y_true, self.y_pred))

    def test_roc_auc_score(self):
        # A simple case for AUC
        y_true_simple = np.array([0, 0, 1, 1])
        y_scores_simple = np.array([0.1, 0.4, 0.35, 0.8])
        self.assertAlmostEqual(roc_auc_score(y_true_simple, y_scores_simple), sklearn_roc_auc(y_true_simple, y_scores_simple))
        
        # A more complex case
        self.assertAlmostEqual(roc_auc_score(self.y_true, self.y_scores), sklearn_roc_auc(self.y_true, self.y_scores))


if __name__ == '__main__':
    unittest.main()
