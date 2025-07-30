"""
Base class for all deep learning feature extraction.
"""
from abc import ABC, abstractmethod
from scipy import ndimage as ndi

class BaseFeatureExtraction(ABC):
    """Base class that all deep learning feature extraction algorithms should inherit from."""
    
    def __init__(self, config=None):
        """
        Initialize the feature extractor.
        
        Args:
            config (dict, optional): Configuration parameters for the extraction class.
        """
        self.config = config or {}
        
    @abstractmethod
    def extract(self, images):
        """
        Segment the input images.
        
        Args:
            List of images (numpy.ndarray with shape NUM IMAGES * HEIGHT * WIDTH * 3): Input images to segment.
            
        Returns:
            numpy.ndarray: Binary mask where 1 indicates the segmented region.
        """
        pass
    
  