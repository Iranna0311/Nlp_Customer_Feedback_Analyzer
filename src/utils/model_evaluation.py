from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def evaluate_classification_model(y_true, y_pred):
    """Compute standard classification metrics and print a report."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    print("\n--- Model Performance Report ---")
    print(classification_report(y_true, y_pred, zero_division=0))
    return metrics