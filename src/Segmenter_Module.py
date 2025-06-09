import torch
from cellpose import models, core, io, plot
from pathlib import Path
import matplotlib.pyplot as plt
# from utils import get_composite 
import numpy as np
from skimage.util import img_as_ubyte

class Segmenter:
    def __init__(self, pretrained_model, 
                 device, data_dir, image_extension,
                 output_dir, offset):     
        io.logger_setup() # Prints progress 
        io.logging.warning("Initializing Segmenter")
        self.config = Config(
            pretrained_model=pretrained_model,
            device=device,
            data_dir=data_dir,
            image_extension=image_extension,
            mask_output_dir=output_dir,
            offset=offset
        )
        # Everything must be correctly initialized before Segmenter can be used
        if core.use_gpu() == False:
            raise ImportError("No GPU access")
        
        if not self.config.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.config.data_dir} does not exist")

        if self.config.pretrained_model is None:
            raise ValueError("Pretrained model must be specified")
      
        self.model = models.CellposeModel(gpu = True, 
                                          pretrained_model=self.config.pretrained_model, 
                                          device=torch.device(self.config.device))

    def load_images(self, image_dir):
        filenames = image_files = io.get_image_files(
            folder=str(image_dir),  # Convert Path object to string
            mask_filter= "_mask",
        ) 
        frames = []
        for image_file in image_files:
            image = io.imread(image_file)
            frames.append(image)

        return frames, filenames
    
    def get_composite(self, dapi, ck, cd45, fitc):
        dtype = dapi.dtype
        max_val = np.iinfo(dapi.dtype).max
        dapi = dapi.astype(np.float32)
        ck = ck.astype(np.float32)
        cd45 = cd45.astype(np.float32)
        fitc = fitc.astype(np.float32)
        rgb = np.zeros((dapi.shape[0], dapi.shape[1], 3),
                    dtype='float')
        rgb[...,0] = ck+fitc
        rgb[...,1] = cd45+fitc
        rgb[...,2] = dapi.astype(np.float32)+fitc
        rgb[rgb > max_val] = max_val
        rgb = rgb.astype(dtype)
        return rgb

    def combine_images(self, images):
        frames=[]
        offset = 10 # SET TO self.config.offset for full data run, or number of sample images (5) for sample run
        for i in range(offset): 
            image0 = images[i]
            image1 = images[i+offset]
            image2 = images[i+2*offset]
            # skip Bright Field scan
            image3 = images[i+3*offset] 

            # FOR DEBUGGING:
            # Visualize the 4 images being before they are combined
            # plt.figure(figsize=(10, 10))
            # plt.subplot(2, 2, 1)
            # plt.imshow(image0, cmap='gray' if len(image0.shape) == 2 else None)
            # plt.title(f"Image {i} (Scan 0)")
            # plt.axis('off')  # Hide axes for better visualization
            # plt.subplot(2, 2, 2)
            # plt.imshow(image1, cmap='gray' if len(image1.shape) == 2 else None)
            # plt.title(f"Image {i+offset} (Scan 1)")
            # plt.axis('off')
            # plt.subplot(2, 2, 3)
            # plt.imshow(image2, cmap='gray' if len(image2.shape) == 2 else None)
            # plt.title(f"Image {i+2*offset} (Scan 2)")
            # plt.axis('off')
            # plt.subplot(2, 2, 4)
            # plt.imshow(image3, cmap='gray' if len(image3.shape) == 2 else None)
            # plt.title(f"Image {i+4*offset} (Scan 3)")
            # plt.axis('off')
            # plt.tight_layout()
            # plt.show()  # Show the images without blocking execution

            final = self.get_composite(image0, image1, image2, image3)
            
            # Show image after combining
            # plt.imshow(final)
            # plt.title(f"Combining Images {i}, {i+offset}, {i+2*offset}, {i+4*offset}")
            # plt.axis('off') 
            
            frames.append(img_as_ubyte(final))
            
        return frames
    
    def segment_frames(self, image_dir):
        print("\nSegmenting frames in directory:", image_dir)
        images, filenames = self.load_images(image_dir) # TODO: Run this on multiple cores

        print("\nCombining 4 scans into 1 image ...")
        frames=self.combine_images(images)

        print("\nComputing masks ...")
        # flush mem per cycle 
        # more sample data, see if models can't find cells in all the slides
        masks, flows, styles = self.model.eval(frames, 
                                               diameter=15,
                                               channels=[0, 0],
                                               )
        
        print("\nSaving the masks ...")
        io.save_masks(
            images=frames,
            masks=masks,
            flows=flows,
            file_names=filenames[0:10], # Set to 5 for sample run, or self.config.offset for full data run
            savedir=self.config.mask_output_dir,
        )
        print("\nMasks saved to:", self.config.mask_output_dir)

        return masks, flows, styles

class Config:
    def __init__(self, pretrained_model, device, data_dir, image_extension, mask_output_dir, offset):
        self.pretrained_model = Path(pretrained_model)
        self.device = device
        self.data_dir = Path(data_dir)
        self.image_extension = image_extension
        self.mask_output_dir = Path(mask_output_dir)
        self.offset = offset