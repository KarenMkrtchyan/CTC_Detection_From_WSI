import numpy as np

def combine_slices(image0, image1, image2, image3):
    """
    Combine four grayscale images into a single 3-channel BGR image.

    :param image0: DAPI
    :param image1: PanCK
    :param image2: CD45/31
    :param image3: VIM

    :return: Combined BGR image as a numpy array.
    """
    brg = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)\

    for i in range(len(brg)):
        for j in range(len(brg[0])):
            brg[i,j,0] = np.uint8(image0[i,j] + image3[i,j]) if image0[i,j] + image3[i,j] < 255 else 255 # hardcoded uint8 max value
            brg[i,j,1] = np.uint8(image2[i,j] + image3[i,j]) if image2[i,j] + image3[i,j] < 255 else 255 
            brg[i,j,2] = np.uint8(image1[i,j] + image3[i,j]) if image1[i,j] + image3[i,j] < 255 else 255 

    return brg

def channels_to_bgr(image, blue_index, green_index, red_index):
    """
    Convert image channels to BGR 3-color format for visualization.
    
    :param image: Input image as numpy array with shape (Height, Width, Channels)
    :param blue_index: List of indices for blue channels in the image.
    :param green_index: List of indices for green channels in the image.
    :param red_index: List of indices for red channels in the image.
    
    """

    # add batch_size as the first dimension of the array 
    if len(image.shape) == 3:
        image = image[np.newaxis, ...]

    # init our future image 
    bgr = np.zeros(
        (image.shape[0], image.shape[1], image.shape[2], 3), dtype="float"
    )

    # combine together different scans (DAPI, PanCK, CD45/31, VIM)
    # into a BGR image
    if len(blue_index) != 0:
        bgr[..., 0] = np.sum(image[..., blue_index], axis=-1)
    if len(green_index) != 0:
        bgr[..., 1] = np.sum(image[..., green_index], axis=-1)
    if len(red_index) != 0:
        bgr[..., 2] = np.sum(image[..., red_index], axis=-1)

    # clip values to the maximum value of the image dtype to avoid downstream overflow
    # bgr is of type float so it has more memmory than the original image and wont itself overflow
    max_val = np.iinfo(image.dtype).max
    bgr[bgr > max_val] = max_val
    bgr = bgr.astype(image.dtype)

    return bgr

def get_composite(dapi, ck, cd45, fitc):
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