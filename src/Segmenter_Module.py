import torch
from cellpose import models, core, io
from pathlib import Path
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt # use in debug console 

class Segmenter:
    def __init__(self, pretrained_model, 
                 device, data_dir, image_extension,
                 output_dir, offset, save_masks):    
        
        io.logger_setup() # Prints progress bar when cellpose is running  
        print("\n🤫 Initializing Segmenter  Module")
        self.config = Config(
            pretrained_model=pretrained_model,
            device=device,
            data_dir=data_dir,
            image_extension=image_extension,
            mask_output_dir=output_dir,
            offset=offset,
            save_masks=save_masks
        )
        # Everything must be correctly initialized before Segmenter can be used
        if core.use_gpu() == False:
            raise ImportError("No GPU access")
        
        if not self.config.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.config.data_dir} does not exist")

        if self.config.pretrained_model is None:
            raise ValueError("Pretrained model must be specified")
            # then load cellpose cpsam
      
        self.model = models.CellposeModel(gpu = True, 
                                          pretrained_model=str(self.config.pretrained_model), 
                                          device=torch.device(self.config.device))

    def load_images(self, image_dir):
        """Load images from the specified directory, and return a list of images as numpy arrays."""
        image_files = sorted(os.listdir(image_dir)) # list index must match the order of scans 

        frames = []
        for image_file in image_files:
            image = cv2.imread(Path(image_dir, image_file), cv2.IMREAD_GRAYSCALE)
            image = (image*257).astype(np.uint16)  # Convert 8-bit to 16-bit
            # image = image.astype(np.float32)
            # image = image * 255 # convert from 8-bit to 16-bit
            # image = image.astype(np.uint16) # convert to 16-bit unsigned integer
            frames.append(image)

        return np.array(frames, dtype=np.uint16) # and if i wasn't clear its 16 bit
    
    def get_composite(self, dapi, ck, cd45, fitc):

        dtype = dapi.dtype
        max_val = np.iinfo(dapi.dtype).max

        dapi = dapi.astype(np.float32)
        ck = ck.astype(np.float32)
        cd45 = cd45.astype(np.float32)
        fitc = fitc.astype(np.float32)

        rgb = np.zeros((dapi.shape[0], dapi.shape[1], 3), dtype='float')
        
        rgb[...,0] = ck+fitc
        rgb[...,1] = cd45+fitc
        rgb[...,2] = dapi.astype(np.float32)+fitc 
        rgb[rgb > max_val] = max_val # Clips overflow 

        rgb = rgb.astype(dtype)
        return rgb
    
    def save_masks(self, masks):
        if not self.config.mask_output_dir.exists():
            self.config.mask_output_dir.mkdir(parents=True, exist_ok=True)
        for i, mask in enumerate(masks):
            mask_path = Path(self.config.mask_output_dir, f"mask_{i}.png")
            cv2.imwrite(mask_path, mask.astype(np.uint16))

    def combine_images(self, images):
        frames=[]
        offset = 10 # SET TO self.config.offset for full data run, or number of sample images (5) for sample run
        for i in range(offset): 
            image0 = images[i]
            image1 = images[i+offset]
            image2 = images[i+2*offset]
            # skip Bright Field scan
            image3 = images[i+3*offset] 
            frames.append(self.get_composite(image0, image1, image2, image3))  

        return frames
    
    def segment_frames(self, frames):
        return self.model.eval(frames,diameter=15,channels=[0, 0]) # test if pasing all the frames at once or one at a time is faster 
    
    def run(self, image_dir):
        print("\n📠 Segmenting frames in directory:", image_dir)
        images = self.load_images(image_dir) # TODO: Run this on multiple cores

        print("\n📠 Combining 4 scans into 1 image ...")
        frames=self.combine_images(images)

        print("\n📠 Computing masks ...")
        masks, flows, styles = self.segment_frames(frames)

        if(self.config.save_masks):
            print("\n📠 Saving the masks ...")
            self.save_masks(masks)
        
        return masks

    def get_cell_crops(self, masks, images):
        """
        Extracts cropped cell images using the segmented masks.

        Arguments:
            masks (np.ndarray): Array of segmented masks with shape (N, H, W).
            images (np.ndarray): Array of original images with shape (N, H, W).

        Returns:
            List[np.ndarray]: List of cropped cell images.
        """

        # plan 
        # loop through image and get the index of all pixels that mattch the current cell instance. 
        # for each cell instance, find the leftmost, rightmost, topmost, bottommost pixels
            # optimization: stop looking after there is a row with no target pixels in it after findind the first 
            # row with pixels 
        # find the leftmost, rightmost, topmost, bottommost pixels of the cell instance
        # find the center using the boundries 
        # go 37.5 pixcels in each direction from the center to get the crop
        # return the crops as a list of numpy arrays
        
        crops = np.array()

        for image in images:
            for i in range(1, np.max(masks)):
                center = self.find_center(masks[i-1], i)
                crop = image[:, center-38:center+36, center-38:center+36]
                crops.append(crop)

        return crops

    def find_center(self, mask, index):
        left = 2000
        right = 0
        top = 2000
        bottom = 0
        for x in range(len(mask)): # TODO make search 'smart' by ending soon and starting later. Could also use pooling to speed up search
            for y in range(len(mask[0])):
                if mask[x,y] == index:
                    if x < left:
                        left = x
                    if x > right:
                        right = x
                    if y < top:
                        top = y
                    if y > bottom:
                        bottom = y

        return ( (left+right)/2, (top+bottom)/2 )
                        

class Config:
    def __init__(self, pretrained_model, device, data_dir, image_extension, mask_output_dir, offset,
                 save_masks):
        self.pretrained_model = Path(pretrained_model)
        self.device = device
        self.data_dir = Path(data_dir)
        self.image_extension = image_extension
        self.mask_output_dir = Path(mask_output_dir)
        self.offset = offset
        self.save_masks = save_masks