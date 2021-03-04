import torch.nn as nn 
import torch.nn.functional as F 

class ReviewClassifier(nn.Module):
    """A simple perceptron-based classifier."""
    def __init__(self, num_features):
        """
        Args:
            num features (int): the size of the input feature vector.
        """
        super(ReviewClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=num_features, out_features=1)

    def forward(self, x_in, apply_sigmoid=False):
        """
        The forward pass of the classifier.
        
        Args:
            x_in (torch.Tensor): An input data tensor. x_in.shape shoudl be (batch, num_features)
            apply_sigmoid (bool): A flag for the sigmoid activation. Should be False if used with the cross-entropy loses.
        Returns:
            The resultig tensor. tensor.shape should be (batch,).
        """

        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = F.sigmoid(y_out)
        return y_out