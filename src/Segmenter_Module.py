import torch
from cellpose import models, core, io
from pathlib import Path
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt # use in debug console
import multiprocessing 

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
    @staticmethod
    def _load_img(args):
        folder, filename = args
        full_path = os.path.join(folder,filename)
        return cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

    def load_images(self, image_dir):
        """Load images from the specified directory, and return a list of images as numpy arrays."""
        image_files = sorted(os.listdir(image_dir)) # list index must match the order of scans 

        with multiprocessing.Pool(multiprocessing.cpu_count() - 2) as p: # save one core for the system and one more for good luck
            args = [(image_dir, f) for f in image_files]
            frames = p.map(Segmenter._load_img, args)

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
        rgb[...,2] = dapi.astype(np.float32)+fitc  # why is there a random np.float32 here?
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
        # set all pixels other than the crop to 0
        # return the crops as a list of numpy arrays

        image_crops = []
        mask_crops = []
        centers = []
        for j in range(len(images)):
            print(f"Processing image {j}/{len(images)}")
            for i in range(1, np.max(masks[j])):
                center = self.find_center(masks[j], i)
                if(center[0] < 38 or center[1] < 38 or center[0] > images[j].shape[1]-38 or center[1] > images[j].shape[2]-38):
                    continue # skip edge cells because they are kind of gross 

                centers.append(center)
                crop = self.crop_img_from_center(center, images[j])
                crop = self.multiplex_mask_on_crop(crop, masks[j], i, center)
                image_crops.append(crop)
                mask_crops.append(self.crop_mask_from_center(center, masks[j]))

        
        return np.array(image_crops), self.binary_masks(mask_crops), np.array(centers) # return the crops and the masks, the masks are used for debugging and visualization purposes only
    
    def multiplex_mask_on_crop(self, crop, mask, index, center): 

        # plan
        # go to top left corner of the mask according to the index
        # loop through the mask and find all pixels that match the index
        # set all pixels that do not match the index to 0 in the crop
        
        for h in range(len(crop)):
            for w in range(len(crop[0])):
                if(mask[h+center[0]-38, w+center[1]-38] != index) and (mask[h+center[0]-37, w+center[1]-37] != index): # there is a sight worry that i'm not matching the crop to mask pixel id perfectly, the extra if statement might be a temp fix 
                    crop[:, h, w] = 0

        return crop

    def crop_img_from_center(self, center, image):
        left = 0 # slighly assymetric, the left gets 38 pixels while the right gets 37 pixels
        right = 75
        bottom = 75
        top = 0
        if(center[0]>38): # Make sure h is not out of range
            if(center[0]<image.shape[1]-38):
                top = center[0] - 38
                bottom = center[0] + 37
            else:
                top = image.shape[1]-75
                bottom = image.shape[1]

        if(center[1]>38): # Make sure w is not out of range
            if(center[1]<image.shape[2]-38):
                left = center[1] - 38
                right = center[1] + 37
            else:
                left = image.shape[2]-75
                right = image.shape[2]
        
        return np.copy(image[:, top:bottom, left:right]) # For images

    def crop_mask_from_center(self, center, image):
        left = 0 # slighly assymetric, the left gets 38 pixels while the right gets 37 pixels
        right = 75
        bottom = 75
        top = 0
        if(center[0]>38): # Make sure h is not out of range
            if(center[0]<image.shape[0]-38):
                top = center[0] - 38
                bottom = center[0] + 37
            else:
                top = image.shape[0]-75
                bottom = image.shape[0]

        if(center[1]>38): # Make sure w is not out of range
            if(center[1]<image.shape[1]-38):
                left = center[1] - 38
                right = center[1] + 37
            else:
                left = image.shape[1]-75
                right = image.shape[1]

        return np.copy(image[top:bottom, left:right]) # For masks

    def find_center(self, mask, index):
        positions = np.argwhere(mask == index)

        if positions.size == 0:
            raise ValueError(f"No pixels found for index {index}")

        top = positions[:, 0].min()
        bottom = positions[:, 0].max()
        left = positions[:, 1].min()
        right = positions[:, 1].max()

        return int((top + bottom) / 2), int((left + right) / 2)

    def binary_masks(self, masks):
        """
        Takes an np.array (shape N, 75, 75) with instance values and converts it to a np.array (shape N, 1, 75, 75)
        with 1 or 0 in the second dimension to indicate mask or no mask.
        Any nonzero value in the original mask is set to 1.
        """
        # Ensure input is a numpy array
        masks = np.asarray(masks)
        # Create binary masks: 1 where mask > 0, else 0
        binary = (masks > 0).astype(np.uint8)
        # Add a channel dimension (axis=1)
        binary = binary[:, np.newaxis, :, :]
        return binary
        

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