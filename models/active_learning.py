import torch
from sklearn.cluster import KMeans

class ActiveLearner:
    def __init__(self, model, n_samples=10):
        self.model = model
        self.n_samples = n_samples

    def select_samples(self, unlabeled_data):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(unlabeled_data)
            uncertainty = torch.var(outputs, dim=1)
            indices = torch.topk(uncertainty, self.n_samples).indices
        return indices.numpy()

    def semi_supervised_update(self, unlabeled_data):
        kmeans = KMeans(n_clusters=2)
        pseudo_labels = kmeans.fit_predict(unlabeled_data.reshape(-1, unlabeled_data.shape[-1]))
        return pseudo_labels
