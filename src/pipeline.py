import yaml
from pathlib import Path
from Segmenter_Module import Segmenter

def main():
    # Load configuration from YAML file
    with open(Path('./src/config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    model = Segmenter(
                      pretrained_model=config['model_dir'], 
                    # pretrained_model="cpsam",
                      device=config['device'], 
                      data_dir=config['data_dir'],
                      image_extension=config['image_extension'],
                      output_dir=config['output_dir'],
                      offset=config['offset'],
                      save_masks=config['save_masks']
                      )
    
    model.segment_frames(Path(config['data_dir']))

if __name__ == "__main__":
    main()

