"""Neural network sentiment model wrappers."""

import os
import joblib
from sklearn.neural_network import MLPClassifier


class NeuralNetworkModel:
	"""Feed-forward neural network classifier for sentiment analysis."""

	def __init__(
		self,
		hidden_layer_sizes=(100,),
		activation="relu",
		solver="adam",
		random_state=42,
		max_iter=300,
	):
		self.model = MLPClassifier(
			hidden_layer_sizes=hidden_layer_sizes,
			activation=activation,
			solver=solver,
			random_state=random_state,
			max_iter=max_iter,
		)

	def train(self, X_train, y_train):
		"""Train the neural network classifier."""
		print("Training Neural Network Model...")
		self.model.fit(X_train, y_train)
		return self

	def predict(self, X_test):
		"""Predict class labels for the given feature matrix."""
		return self.model.predict(X_test)

	def predict_proba(self, X_test):
		"""Predict class probabilities for the given feature matrix."""
		return self.model.predict_proba(X_test)

	def score(self, X_test, y_test):
		"""Return the classifier accuracy."""
		return self.model.score(X_test, y_test)

	def save_model(self, filepath: str):
		"""Persist the trained neural network to disk."""
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		joblib.dump(self.model, filepath)
		print(f"Neural network model saved to: {filepath}")

	def load_model(self, filepath: str):
		"""Load a saved neural network model from disk."""
		if not os.path.exists(filepath):
			raise FileNotFoundError(f"No model found at {filepath}")

		self.model = joblib.load(filepath)
		print(f"Neural network model loaded from: {filepath}")