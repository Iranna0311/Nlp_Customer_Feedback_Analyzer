"""Support vector machine wrapper for sentiment classification."""

from sklearn.svm import SVC
import numpy as np


class SVMClassifierModel:
    """Support Vector Classifier model for sentiment analysis."""
    
    def __init__(self, kernel='linear', C=1.0, random_state=42):
        self.kernel = kernel
        self.C = C
        self.random_state = random_state
        self.model = SVC(kernel=kernel, C=C, random_state=random_state, probability=True)
    
    def train(self, X_train, y_train):
        """Train the SVM model.
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels
        """
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        """Make predictions on test data.
        
        Args:
            X_test: Test feature matrix
            
        Returns:
            Predicted labels
        """
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities.
        
        Args:
            X_test: Test feature matrix
            
        Returns:
            Prediction probabilities
        """
        return self.model.predict_proba(X_test)
    
    def score(self, X_test, y_test):
        """Calculate model accuracy.
        
        Args:
            X_test: Test feature matrix
            y_test: Test labels
            
        Returns:
            Accuracy score
        """
        return self.model.score(X_test, y_test)