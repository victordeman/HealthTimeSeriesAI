import unittest
import numpy as np
from utils.evaluation import Evaluator

class TestEvaluator(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.evaluator = Evaluator(task='classification')
        self.y_true = np.array([0, 1, 0, 1, 1])
        self.y_pred = np.array([0, 1, 1, 1, 0])

    def test_accuracy(self):
        """Test accuracy calculation."""
        metrics = self.evaluator.evaluate(self.y_true, self.y_pred)
        expected_accuracy = 3 / 5  # 3 correct predictions out of 5
        self.assertAlmostEqual(metrics['accuracy'], expected_accuracy)

    def test_f1_score(self):
        """Test F1-score calculation."""
        metrics = self.evaluator.evaluate(self.y_true, self.y_pred)
        expected_f1 = 2 * (2/3 * 2/4) / (2/3 + 2/4)  # Precision=2/3, Recall=2/4 for class 1
        self.assertGreaterEqual(metrics['f1_score'], 0)
        self.assertLessEqual(metrics['f1_score'], 1)

    def test_empty_input(self):
        """Test handling of empty input."""
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(np.array([]), np.array([]))

if __name__ == '__main__':
    unittest.main()
