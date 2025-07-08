import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class CustomImageDataset(Dataset):
    """
    Custom dataset class for loading and processing 4-channel, 75x75, 16-bit TIFF images.
    Each sample includes a normalized image, a binary mask, and a label. The dataset
    supports applying transformations during data loading.

    Attributes:
        images (np.ndarray): Numpy array of images with shape (N, 4, 75, 75).
        masks (np.ndarray): Numpy array of binary masks with shape (N, 1, 75, 75).
        labels (np.ndarray): Numpy array of labels corresponding to each image.
        tran (bool): Flag to indicate if transformations should be applied.
        t (torchvision.transforms.Compose): Transformation pipeline to convert numpy arrays to tensors.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Retrieves the image, mask, and label at the given index.
    """

    def __init__(self, images, masks, labels, tran=False, offset=2304):
        """
        Initializes the custom image dataset.

        Parameters:
            images (np.ndarray): Array of input images.
            masks (np.ndarray): Array of binary masks corresponding to the images.
            labels (np.ndarray): Array of integer labels for each image.
            tran (bool): If True, applies transformations during data loading.
        """

        # image - > 5 channels: dapi, ck, cd45, fitc, mask with shape  (N, 4, 75, 75).
        self.images = images        
        self.masks =  masks  
        self.labels = labels
        self.tran = tran

        # Transformation pipeline: Convert numpy arrays to PyTorch tensors
        self.t = transforms.Compose([
            transforms.ToTensor()  # Converts H*W*C numpy array to C*H*W tensor
        ])

        # self.t = transforms.Compose([
        #     torch.from_numpy() # Woudlnt need to flip the image before passing but gives an error
        # ])

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieves the image, mask, and label at the given index.

        Parameters:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - hard_masked_image: The image multiplied by the mask, concatenated with the mask.
                - label: The label corresponding to the image.
        """
        # Normalize the image to the range [0, 1]
        image = self.images[idx].astype(np.float32) / 65535.0
        # image = np.transpose(image, (2, 0, 1)) # ruin a perfectly good H W C to C H W so it can be moved back later 

        # Retrieve the corresponding label and mask
        label = self.labels[idx]
        mask = self.masks[idx].astype(np.int16)
        # mask = np.transpose(mask, (2,0,1))

        # Apply transformation to convert to tensor
        # image = self.t(image)
        # mask = self.t(mask)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        
        # Create a masked version of the image by multiplying with the binary mask
        hard_masked_image = image * mask

        # Concatenate the masked image and the mask itself along the channel dimension
        hard_masked_image = torch.cat((hard_masked_image, mask), dim=0)

        return hard_masked_image, torch.tensor(label, dtype=torch.long)
