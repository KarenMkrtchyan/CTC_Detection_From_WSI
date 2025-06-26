import yaml
from pathlib import Path
from Segmenter_Module import Segmenter
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
                      save_masks=config['save_masks']
                      )
    
    extraction_model = Extractor(
        model_path=config['extraction_model'],
        device=config['device']
    )
    
    # segmentor_model.run(Path(config['data_dir']))
    
    print("\n📠 Segmenting frames in directory:")
    images = segmentor_model.load_images(Path(config['data_dir'])) # TODO: Run this on multiple cores

    print("\n📠 Combining 4 scans into 1 image ...")
    frames=segmentor_model.combine_images(images)

    print("\n📠 Computing masks ...")
    masks, flows, styles = segmentor_model.segment_frames(frames)

    print("\n📠 Doing all the data loader nonsense ...")
    # image - > 5 channels: dapi, ck, cd45, fitc, mask 
    dataset = CustomImageDataset(images, masks, labels=np.zeros(images.shape[0]), tran=False)
    dataloader = DataLoader(dataset, batch_size=config['inference_batch'], shuffle=False)

    print("\n📠 Extracting Features ...")
    embeddings = extraction_model.get_embeddings(dataloader)
    embeddings = embeddings.numpy()

    embeddings_df = pd.DataFrame(
        embeddings.astype('float16'),
        columns=[f'z{i}' for i in range(embeddings.shape[1])])
    features = pd.concat([features, embeddings_df], axis=1)

    print("Its over")

if __name__ == "__main__":
    main()

