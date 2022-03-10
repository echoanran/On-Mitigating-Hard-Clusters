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

## Dataset Preparation

Here we use MS1M dataset as an example.

### Data format
The data directory is constucted as follows:
```
.
├── data
|   ├── features
|   |   └── xxx.bin
│   ├── labels
|   |   └── xxx.meta
│   ├── knns
|   |   └── ... 
```

- `features` currently supports binary file.
- `labels` supports plain text where each line indicates a label corresponding to the feature file.
- `knns` can also be computed with `is_reload` in configuration files set to True.

Take MS1M (Part0 and Part1) as an example. The data directory is as follows:
```
data
  ├── features
    ├── part0_train.bin                 # acbbc780948e7bfaaee093ef9fce2ccb
    ├── part1_test.bin                  # ced42d80046d75ead82ae5c2cdfba621
  ├── labels
    ├── part0_train.meta                # class_num=8573, inst_num=576494
    ├── part1_test.meta                 # class_num=8573, inst_num=584013
  ├── knns
    ├── part0_train/faiss_k_80.npz      # 5e4f6c06daf8d29c9b940a851f28a925
    ├── part1_test/faiss_k_80.npz       # d4a7f95b09f80b0167d893f2ca0f5be5
```

### Downloads
- [MS1M](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/)
    - Part1 (584K): [GoogleDrive](https://drive.google.com/open?id=16WD4orcF9dqjNPLzST2U3maDh2cpzxAY).
    - Other Test Subsets (5.21M): [GoogleDrive](https://drive.google.com/file/d/10boLBiYq-6wKC_N_71unlMyNrimRjpVa/view?usp=sharing).
    - Precomputed KNN: [GoogleDrive](https://drive.google.com/file/d/1CRwzy899vkLqIYm60AzDsaDEBuwgxNlY/view?usp=sharing).

## Configuration

Configuration files are provided in `./config`. 
- `config_train_ms1m.yaml` for training our similarity prediction model on the training set, i.e., "part0_train". 
- `config_eval_ms1m_part*.yaml` for evaluation on the 5 test subsets, i.e., "part1_test", "part3_test", "part5_test", "part7_test", "part9_test".

## Training

After setting the configuration, to start training, simply run

```bash
python main.py -c ./config/config_train_ms1m.yaml
```

Folder for saving checkpoints is specified in the configuration file using parameter `work_dir`.

We provide a pre-trained model `checkpoint.tar` in `./save/Ours`.

## Test
Once the training is completed, the obtained model can be used for clustering. To start clustering on the test subset "part*_test", simply run

``` bash
python eval.py -c ./config/config_eval_ms1m_part*.yaml
```

The clustering results will be saved in `work_dir/results`.