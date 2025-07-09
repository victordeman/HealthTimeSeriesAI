import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, seq_len)

    def forward(self, z):
        h, _ = self.lstm(z)
        return torch.sigmoid(self.linear(h))

class Discriminator(nn.Module):
    def __init__(self, seq_len, hidden_dim):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(seq_len, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return torch.sigmoid(self.linear(h))

class TimeSeriesSynthesizer:
    def __init__(self, seq_len=187, input_dim=187, hidden_dim=64, sparsity_prob=0.1):
        self.generator = Generator(input_dim, hidden_dim, seq_len)
        self.discriminator = Discriminator(seq_len, hidden_dim)
        self.seq_len = seq_len
        self.sparsity_prob = sparsity_prob
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

    def train(self, data, epochs=50):
        data = torch.tensor(data, dtype=torch.float32)
        batch_size = 32
        for epoch in range(epochs):
            for i in range(0, len(data), batch_size):
                real_data = data[i:i+batch_size]
                real_labels = torch.ones(real_data.size(0), 1)
                fake_labels = torch.zeros(real_data.size(0), 1)
                self.d_optimizer.zero_grad()
                real_output = self.discriminator(real_data)
                d_loss_real = self.criterion(real_output, real_labels)
                z = torch.randn(real_data.size(0), self.seq_len, self.seq_len)
                fake_data = self.generator(z)
                fake_output = self.discriminator(fake_data.detach())
                d_loss_fake = self.criterion(fake_output, fake_labels)
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()
                self.g_optimizer.zero_grad()
                fake_output = self.discriminator(fake_data)
                g_loss = self.criterion(fake_output, real_labels)
                g_loss.backward()
                self.g_optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

    def generate(self, n_samples):
        z = torch.randn(n_samples, self.seq_len, self.seq_len)
        with torch.no_grad():
            synthetic_data = self.generator(z).numpy()
        mask = np.random.binomial(1, 1 - self.sparsity_prob, synthetic_data.shape)
        synthetic_data = synthetic_data * mask
        return synthetic_data

def generate_synthetic_variants(input_file, output_file, n_samples=100):
    df = pd.read_csv(input_file)
    X = df.iloc[:, :-1].values
    synthesizer = TimeSeriesSynthesizer(seq_len=X.shape[1])
    synthesizer.train(X, epochs=50)
    synthetic_X = synthesizer.generate(n_samples)
    synthetic_labels = np.random.randint(0, 2, n_samples)  # Random labels for demo
    synthetic_df = pd.DataFrame(np.hstack([synthetic_X, synthetic_labels.reshape(-1, 1)]),
                                columns=df.columns)
    synthetic_df.to_csv(output_file, index=False)
