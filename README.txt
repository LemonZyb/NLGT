# unzip the ogbn-arxiv dataset

unzip arxiv.zip

# setup the environment
conda create -n NLGT python=3.10
conda activate NLGT
pip install -r requirements 

# Hierarchical Neighborhood Sampling for each node
python sampling.py

# train with default parameter
python train.py

# evaluate your model
python evaluate.py