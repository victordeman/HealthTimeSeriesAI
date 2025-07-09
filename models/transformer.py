import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, n_heads=4, n_layers=2, dropout=0.1, n_classes=2):
        super(TimeSeriesTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=256, dropout=dropout),
            num_layers=n_layers
        )
        self.output_layer = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        x = self.output_layer(x[:, -1, :])
        return x
