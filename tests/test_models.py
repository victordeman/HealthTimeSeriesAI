import unittest
import torch
import numpy as np
from models.transformer import TimeSeriesTransformer
from models.stream_learner import StreamLearner
from models.active_learning import ActiveLearner
from models.glassbox import GlassboxExplainer

class TestModels(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.input_dim = 1
        self.seq_len = 187
        self.batch_size = 2
        self.model = TimeSeriesTransformer(input_dim=self.input_dim, n_classes=2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.sample_data = torch.randn(self.batch_size, self.seq_len, self.input_dim)

    def test_transformer_forward(self):
        """Test transformer forward pass."""
        output = self.model(self.sample_data)
        self.assertEqual(output.shape, (self.batch_size, 2))  # Expect (batch_size, n_classes)

    def test_stream_learner(self):
        """Test stream learner update."""
        stream_learner = StreamLearner(self.model, self.optimizer, batch_size=2)
        stream_learner.update(self.sample_data[0:1])
        self.assertTrue(len(stream_learner.buffer) == 1)

    def test_active_learner(self):
        """Test active learner sample selection."""
        active_learner = ActiveLearner(self.model, n_samples=1)
        indices = active_learner.select_samples(self.sample_data)
        self.assertEqual(len(indices), 1)

    def test_glassbox_explainer(self):
        """Test glassbox explainer attention output."""
        explainer = GlassboxExplainer(self.model)
        attention = explainer.explain(self.sample_data)
        self.assertEqual(attention.shape, (self.batch_size, self.seq_len, self.seq_len))

if __name__ == '__main__':
    unittest.main()
