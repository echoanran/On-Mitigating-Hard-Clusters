# On Mitigating Hard Clusters for Face Clustering

# Dependency

- python>=3.6
- pytorch>=1.6.0
- torchvision>=0.8.1

```bash
conda install faiss-gpu -c pytorch
pip install -r requirements.txt
```

# Usage

## Configuration

Configuration files are provided in "./config".

"config_train_ms1m.yaml" for training our similarity prediction model on the training set, i.e., "part0_train".

"config_eval_ms1m_part*.yaml" for evaluation on the 5 test subsets, i.e., "part1_test", "part3_test", "part5_test", "part7_test", "part9_test".

## Training

After setting the configuration, to start training, simply run

> python main.py -c ./config/config_train_ms1m.yaml

Folder for saving checkpoints is specified in the configuration file using parameter "work_dir".
We provide a pre-trained model "checkpoint.tar" in "./save/Ours".

## Test
Once the training is completed, we can use the model for clustering.
To start clustering on the test subset "part*_test", simply run

> python eval.py -c ./config/config_eval_ms1m_part*.yaml

The clustering results will be saved in "work_dir/results".

# Dataset

Refer to https://github.com/yl-1993/learn-to-cluster/blob/master/DATASET.md