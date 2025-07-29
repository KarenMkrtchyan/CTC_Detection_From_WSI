import torch
from cellpose import models, core, io
from pathlib import Path
import os
import numpy as np
import cv2
import multiprocessing 
from segmentation_module.Base import BaseSegmenter
from segmentation_module.utils.config import Config
from segmentation_module.utils.loader import load_img
from segmentation_module.utils.image import compute_composite
from segmentation_module.utils.crop import find_center, crop_img_from_center, multiplex_mask_on_crop, crop_mask_from_center, crop_single_image
from segmentation_module.utils.mask import binary_masks

class Segmenter(BaseSegmenter):
    """
    The Segmenter class handles image segmentation using a Cellpose model with GPU acceleration.

    This class provides functionality for:
    - Loading grayscale microscopy images from a directory
    - Combining multi-channel scans into composite images
    - Running segmentation using Cellpose
    - Saving mask outputs
    - Extracting cropped cell images from masks

    Attributes:
        model (cellpose.models.CellposeModel): The loaded Cellpose model.
        config (Config): Configuration object containing paths and settings.

    Methods:
        load_images(image_dir): Loads images from a directory using multiprocessing.
        combine_images(images): Combines 4-channel scans into RGB composites.
        segment_frames(frames): Runs Cellpose segmentation on image frames.
        save_masks(masks): Saves the predicted masks to disk.
        get_cell_crops(masks, images): Extracts cropped cell images and their masks.
        run(image_dir): Main workflow to segment images from a directory.
    """
    def __init__(self, pretrained_model, 
                 device, data_dir, image_extension,
                 output_dir, offset):    
        
        io.logger_setup() # Prints progress bar when cellpose is running  
        print("\n🤫 Initializing Segmenter  Module")
        self.config = Config(
            pretrained_model=pretrained_model,
            device=device,
            data_dir=Path(data_dir),
            image_extension=image_extension,
            mask_output_dir=Path(output_dir),
            offset=offset,
        )
        if core.use_gpu() == False:
            raise ImportError("No GPU access")
        
        if not self.config.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.config.data_dir} does not exist")

        if self.config.pretrained_model is None:
            raise ValueError("Pretrained model must be specified")
      
        self.model = models.CellposeModel(gpu = True, 
                                          pretrained_model=str(self.config.pretrained_model), 
                                          device=torch.device(self.config.device))
        self.image_data = np.empty(1)
        self.composite_data = np.empty(1)
        self.masks = np.empty(1)

    def segment(self, images=None):
        """
        Segment the input images.
        
        Args:
            List of images (numpy.ndarray with shape NUM IMAGES * HEIGHT * WIDTH * 3): Input images to segment.
            
        Returns:
            numpy.ndarray: Insance mask where each cell gets its own ID
        """
        if not images:
            images = self.composite_data

        self.masks, _, _ = self.model.eval(self.composite_data, diameter=15, channels=[0, 0]) 
        return self.masks
        
    def preprocess(self, images=None):
        """
        Preprocess the loaded input images before segmentation by combining different scan types into a BRG image understood by the segmentation module.
        
        Args:
            image (numpy.ndarray): Input image to preprocess.
            
        Returns:
            numpy.ndarray: Preprocessed image.
        """
        if not images:
            images = self.image_data

        frames=[]
        offset = int(len(images)/4) 
        for i in range(offset): 
            image0 = images[i]
            image1 = images[i+offset]
            image2 = images[i+2*offset]
            # skip Bright Field scan
            image3 = images[i+3*offset] 
            frames.append(compute_composite(image0, image1, image2, image3))  

        self.composite_data = frames
        return frames 

    def postprocess(self, masks=None, images=None):
        """
        Postprocess the segmentation mask. Extracts cropped cell images using the segmented masks.
        
 
        Arguments:
            masks (np.ndarray): Array of segmented masks with shape (N, H, W).
            images (np.ndarray): Array of original images with shape (N, H, W).
        Returns:
            List[np.ndarray]: List of cropped cell images.
        """
        if not masks:
            masks = self.masks
        if not images:
            images = self.composite_data

        # multiprocessing.set_start_method('spawn', force=True)

        # Prepare arguments for each image/mask pair
        args = [
            (
                masks[j], images[j],
                find_center,
                crop_img_from_center,
                multiplex_mask_on_crop,
                crop_mask_from_center
            )
            for j in range(len(images))
        ]

        with multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() - 2)) as pool:
            results = pool.map(crop_single_image, args)

        # Flatten results
        image_crops, mask_crops, centers = [], [], []
        for img_crops, msk_crops, ctrs in results:
            image_crops.extend(img_crops)
            mask_crops.extend(msk_crops)
            centers.extend(ctrs)

        del self.image_data
        del self.composite_data
        del self.masks

        return (
            np.stack(image_crops, axis = 0),
            np.array(binary_masks(np.stack(mask_crops), axis=0)),
            np.stack(centers, axis=0)
        )

    def load_data(self, image_dir):
        """
        Load images from the specified directory, and return a list of images as numpy arrays.
        The returned value is optional to use and self.image_data is what the segment wants to use unless overwritten
        
        Args:
            image_dir(Path): os-valid path (Use pathlib.Path) for the folder with slide data
        
        """
        image_files = sorted(os.listdir(image_dir)) # list index must match the order of scans 

        with multiprocessing.Pool(multiprocessing.cpu_count() - 2) as p: # save one core for the system and one more for good luck
            args = [(image_dir, f) for f in image_files]
            frames = p.map(load_img, args)

        self.image_data = np.array(frames, dtype=np.uint16) 
        return self.image_data 

    def save_masks(self, masks):
        if not self.config.mask_output_dir.exists():
            self.config.mask_output_dir.mkdir(parents=True, exist_ok=True)
        for i, mask in enumerate(masks):
            mask_path = Path(self.config.mask_output_dir, f"mask_{i}.png")
            cv2.imwrite(mask_path, mask.astype(np.uint16))

