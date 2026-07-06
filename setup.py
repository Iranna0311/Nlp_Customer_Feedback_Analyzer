from setuptools import setup, find_packages

setup(
    name="nlp_customer_feedback_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "nltk",
        "spacy",
    ],
)