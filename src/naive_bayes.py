import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        self.class_priors = self.estimate_class_priors(labels)
        self.vocab_size = features.shape[1]  
        self.conditional_probabilities = self.estimate_conditional_probabilities(features, labels, delta)

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        labels = labels.to(torch.int64) 
        class_counts = torch.bincount(labels)
        total_samples = labels.shape[0]
        class_priors = {i: torch.tensor([class_counts[i] / total_samples]) for i in range(len(class_counts))}
        return class_priors


    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """
        unique_classes = torch.unique(labels)
        v = features.shape[1] 


        class_word_probs = {}

        for c in unique_classes:
            class_features = features[labels == c]
            word_counts = class_features.sum(dim=0)
            total_words = word_counts.sum() + delta * v 
            class_word_probs[c.item()] = (word_counts + delta) / total_words

        return class_word_probs

    def estimate_class_posteriors(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError("Model must be trained before estimating class posteriors.")

        log_posteriors = torch.zeros(len(self.class_priors))

        for c in self.class_priors.keys():
            log_prior = torch.log(self.class_priors[c])
            log_likelihood = (torch.log(self.conditional_probabilities[c]) * feature).sum()
            log_posteriors[c] = log_prior + log_likelihood  

        return log_posteriors

    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).
        """
        log_posteriors = self.estimate_class_posteriors(feature)
        return torch.argmax(log_posteriors).item()

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.
        """
        log_posteriors = self.estimate_class_posteriors(feature)
        probs = torch.nn.functional.softmax(log_posteriors, dim=0)  
        return probs
