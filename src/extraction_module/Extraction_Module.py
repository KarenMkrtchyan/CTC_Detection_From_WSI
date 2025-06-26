import torch
from .Contrastive_Learning import CL

class Extractor:
    def __init__(self, model_path, device='cpu'):
        print("\n📠 Initializing the feature extractor ...")
        if model_path is None:
            raise ValueError("Model must be provided for feature extraction.")
        
        self.device = device
        self.model = CL(in_channels=5, h_dim=128, projection_dim=64)  # Instantiate the model
        state_dict = torch.load(model_path, map_location=self.device)['model_state_dict']
        self.model.load_state_dict(state_dict) 
        self.model.eval()

    
    def get_embeddings(self, dataloader):
        """
        Extract embeddings from the encoder model for a given dataset.

        Parameters:
            dataloader (torch.utils.DataLoader): DataLoader object to provide batches of images.

        Returns:
            torch.Tensor: Concatenated embeddings from the entire dataset.
        """
        self.model.eval()  # Set model to evaluation mode
        embeddings = []

        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)  # Move data to device
                embeddings.append(self.model.encoder(x).detach().cpu())  # Store embeddings 
        # Concatenate all embedding tensors
        embeddings = torch.cat(embeddings)
        return embeddings
