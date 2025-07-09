from sklearn.metrics import f1_score, accuracy_score

class Evaluator:
    def __init__(self, task='classification'):
        self.task = task

    def evaluate(self, y_true, y_pred):
        if self.task == 'classification':
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'f1_score': f1_score(y_true, y_pred, average='weighted')
            }
        return {}
