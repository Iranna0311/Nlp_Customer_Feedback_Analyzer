"""Model wrappers for customer feedback analysis."""

from importlib import import_module

__all__ = [
	"LogisticRegressionModel",
	"NeuralNetworkModel",
	"RandomForestModel",
	"SVMClassifierModel",
]


def __getattr__(name):
	module_map = {
		"LogisticRegressionModel": "src.models.logistic_regression",
		"NeuralNetworkModel": "src.models.neural_networks",
		"RandomForestModel": "src.models.random_forest",
		"SVMClassifierModel": "src.models.svm",
	}

	if name not in module_map:
		raise AttributeError(f"module 'src.models' has no attribute '{name}'")

	module = import_module(module_map[name])
	return getattr(module, name)