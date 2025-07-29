import yaml
from pathlib import Path
from segmentation_module.Segmenter import Segmenter
from extraction_module.Extraction_Module import Extractor
from extraction_module.Data_Handler import CustomImageDataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


def main():
    with open(Path('./src/config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
        
    segmentor_model = Segmenter(
                      pretrained_model=config['segmentation_model'], 
                      device=config['device'], 
                       data_dir=config['data_dir'],
                      image_extension=config['image_extension'],
                      output_dir=config['output_dir'],
                      offset=config['offset'],
                      )   
        
    print("\n📠 Segmenting frames in directory: ", config['data_dir'])
    segmentor_model.load_data(Path(config['data_dir'])) # TODO: Run this on multiple cores
    print("\n📠 Combining 4 scans into 1 image ...")
    segmentor_model.preprocess()
    print("\n📠 Computing masks ...")
    segmentor_model.segment()
    print("\n📠 Cropping images ...")
    image_crops, mask_crops, centers = segmentor_model.postprocess()
    del segmentor_model

    print("\n📠 Preping segmentation output for extraction input ...")
    dataset = CustomImageDataset(image_crops, mask_crops, labels=np.zeros(image_crops.shape[0]), tran=False)
    dataloader = DataLoader(dataset, batch_size=config['inference_batch'], shuffle=False)

    extraction_model = Extractor(
        model_path=config['extraction_model'],
        device=config['device']
    )
    
    print("\n📠 Extracting Features ...")
    embeddings = extraction_model.get_embeddings(dataloader)
    embeddings_np = embeddings.cpu().numpy()

    print("\n📠 Saving Features ...")
    embeddings_df = pd.DataFrame(
        embeddings_np.astype('float16'),
        columns=[f'z{i}' for i in range(embeddings.shape[1])])
    
    embeddings_df.insert(0, "slide id", 0)
    embeddings_df.insert(1, "center_x", centers[:, 0])
    embeddings_df.insert(2, "center_y", centers[:, 1])

    embeddings_df.to_parquet("data/processed/embeddings.parquet.gzip", compression="gzip")

    # now feed this to event characterization - Rafael 
    # how to set up repo (strucutre/filenames) traditional and deep learning modules
    # so we dont interferece with each other 
    # abc programing folder structure, naming, cli commands,
    # rest of the phd students for desktop dlx amin calendar-supports 3 ppl at a time 
       
    # abstract base class to classifer and shi but lowkey kinda fried if u ask me 
    # init function
    # prediction -> argmax
    # proabilites -> array

    # pass in df and get an array of outputs 
    # load multiple cellpose models into vram mem 32 threads 
    # make it blazing fast 

    # or use like a million threads to load all of the inputs into the model in one shot and take 5 seconds but need to fuck with cellpose code in that case


    print("\nPipeline finished\n")

if __name__ == "__main__":
    main()