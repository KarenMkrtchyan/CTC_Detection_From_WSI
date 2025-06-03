import numpy as np
import torch
from cellpose import models, core, io, plot
from pathlib import Path
import os
import matplotlib.pyplot as plt

class Segmenter:
    def __init__(self, pretrained_model, 
                 device, data_dir, image_extension,
                 output_dir):     
        io.logger_setup() # Prints progress 
        io.logging.warning("Initializing Segmenter with configuration:")
        self.config = Config(
            pretrained_model=pretrained_model,
            device=device,
            data_dir=data_dir,
            image_extension=image_extension,
            mask_output_dir=output_dir
        )
        # Everything must be correctly initialized before Segmenter can be used
        if core.use_gpu() == False:
            raise ImportError("No GPU access")
        
        if not self.config.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.config.data_dir} does not exist")

        if self.config.pretrained_model is None:
            raise ValueError("Pretrained model must be specified")
      
        self.model = models.CellposeModel(gpu = True, pretrained_model=self.config.pretrained_model, device=torch.device(self.config.device))

    def segment_frame(self, img):
        return self.model.eval(img, diameter=None, channels=[0, 0])

    def load_images(self, image_dir):
        filenames = image_files = io.get_image_files(
            folder=str(image_dir),  # Convert Path object to string
            mask_filter= "_mask",
        ) 
        print("FILENAMES:", filenames)
        frames = []
        for image_file in image_files:
            image = io.imread(image_file)
            frames.append(image)

            # Display the image for debugging
            plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
            plt.title(f"Loaded Image: {image_file}")
            plt.axis('off')  # Hide axes for better visualization
            plt.show()  # Show the image without blocking execution
        print("FRAMES: "   , frames)
        return frames, filenames

    def segment_frames(self, image_dir):
        print("Segmenting frames in directory:", image_dir)
        frames, filenames = self.load_images(image_dir)
        masks, flows, styles = self.model.eval(frames, 
                                               diameter=None, 
                                               channels=[0, 0], 
                                               augment=False)
        io.save_masks(
            images=frames,
            masks=masks,
            flows=flows,
            file_names=filenames,
        )

        return masks, flows, styles



class Config:
    def __init__(self, pretrained_model, device, data_dir, image_extension, mask_output_dir):
        self.pretrained_model = pretrained_model
        self.device = device
        self.data_dir = Path(data_dir)
        self.image_extension = image_extension
        self.mask_output_dir = mask_output_dir