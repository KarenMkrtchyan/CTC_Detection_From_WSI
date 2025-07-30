import torch
from ..Base import BaseAnaliser
from .MLP import MLP
import numpy as np

class SpikeIn(BaseAnaliser):
    def __init__(self, model_path):
        self.model = MLP(in_dim=128, h_dim=32, out_dim=3) 
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def prediction(self, data):
        """
        Class that will return the argmax of the final layer of the model from a single 
        cell embedding .

        Args:
            data (np.ndarray): vector representing a cell from feature extraction models

        Returns:
           Output of Nerual net model
        """
        if isinstance(data, np.ndarray) and data.shape == (128,):
            with torch.no_grad():
                input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # shape (1, 128)
                output = self.model(input_tensor)
                predicted_class = torch.argmax(output, dim=1).item()

            return predicted_class

        else:
            raise ValueError("Input must be a 128 dim vector")
        
    def probability(self, data):
        """
        Class that will return the softmax of the final layer of the model from a single 
        cell embedding .

        Args:
            data (np.ndarray): vector representing a cell from feature extraction models

        Returns:
            numpy.ndarray: vecotr with probabilitets of different events
        """
        if isinstance(data, np.ndarray) and data.shape == (128,):
            with torch.no_grad():
                input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # shape (1, 128)
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1).squeeze(0).numpy()

            return probabilities
        else:
            raise ValueError("Input must be a 128 dim vector")
