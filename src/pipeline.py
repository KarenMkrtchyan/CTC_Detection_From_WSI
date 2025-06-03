import yaml
from pathlib import Path
from Segmenter_Module import Segmenter

def main():
    # Load configuration from YAML file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    image_dir = Path('./data/images')
    model = Segmenter(pretrained_model=config['model_dir'], 
                      device=config['device'], 
                      data_dir=config['data_dir'],
                      image_extension=config['image_extension'],
                      output_dir=config['output_dir']
                      )
    model.segment_frames(image_dir)

if __name__ == "__main__":
    main()

