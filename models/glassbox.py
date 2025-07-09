import matplotlib.pyplot as plt

class GlassboxExplainer:
    def __init__(self, transformer_model):
        self.model = transformer_model

    def explain(self, input_data):
        self.model.eval()
        with torch.no_grad():
            input_proj = self.model.input_projection(input_data)
            attention = self.model.transformer.layers[-1].self_attn(input_proj, input_proj, input_proj)[1]
        return attention

    def visualize_attention(self, attention, feature_names, filename='attention.png'):
        plt.figure(figsize=(10, 8))
        plt.imshow(attention[0].cpu().numpy(), cmap='viridis')
        plt.title("Attention Weights for Time Series")
        plt.xlabel("Time Steps")
        plt.ylabel("Features")
        plt.xticks(range(len(feature_names)), feature_names, rotation=45)
        plt.colorbar()
        plt.savefig(filename)
        plt.close()
