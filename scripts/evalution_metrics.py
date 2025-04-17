from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Flatten for comparison
    labels = labels.flatten()
    predictions = predictions.flatten()

    # Mask out padding tokens (label = -100 in HF datasets)
    mask = labels != -100
    labels = labels[mask]
    predictions = predictions[mask]

    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted'),
    }
