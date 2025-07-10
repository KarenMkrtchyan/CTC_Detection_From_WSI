# ML pipeline prototypes for whole slide microscopty images 
Pipeline for loading in microscopy images, segmenting them, and extracting cell features into embeddings for downstream tasks

## Conda env
conda create --name cellpose python=3.10
conda activate cellpose

## Install reqs
python -m pip install -r requirements.txt 

## Run
cd to root directory
update config.yaml as needed
in pipeline.py, if using sample data, the offset should be 10 (or whatever number of sample images is used)
run `python pipeline.py`
check data/processed for embeddings for each cell found in the slide