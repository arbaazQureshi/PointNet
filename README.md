#### Table of contents
1. [Getting Started](#Getting-Started)
2. [Experiments](#Experiments)
3. [Acknowledgments](#Contributions)

# Self-Supervised Pre-Training for 3D Point Networks with Multi-View Rendering
674 class project

Siddhant Shingi,
Nidhi Chandra,
Arbaaz Qureshi,
<br>
University of Massachusetts, Amherst

## Getting Started

### Installation

- Download the zip file, unzip to get the code folder
```bash
cd self_sup_PC_repr_learning
```

- Install dependencies:
```bash
conda env create -f environment.yml
conda activate sspcd
Download code from https://github.com/NVIDIA/MinkowskiEngine/releases/tag/v0.5.0, compile and install MinkowskiEngine.
```

### Datasets

- **Download the dataset from [here](https://drive.google.com/drive/folders/1tHHgeX50e5YhdruxTimWCcWXhkpIOPOo)**

## Experiments
### Pre-trained Models.
We also provide pretrained models in `pretrain_mod/runs/<model_name>`

Model names:
* ws_8: window size 8 * 8, resnet50 backbone frozen 
* ws_16: window size 16 * 16, resnet50 backbone frozen 
* resnet_ws_8: window size 8 * 8, resnet50 backbone unfrozen 
* resnet_ws_16: window size 16 * 16, resnet50 backbone unfrozen 


### Pre-training
```bash
cd pretrain_mod #  to pretrain with our approach
python train.py \
--num_views 12 \ # number of view used for each object
--num_point_contrast 512 \ # number of positive pair for contrastive loss
--num_points 1024 \ # number of points for each object
--dataset path_to_folder_dataset \
--log_dir path_to_result_model \
--path_model path_to_pre_trained_2d_image \
--model pointnet \ # pre-training backbone pointnet or dgcnn
--window_size=16 \ # window size
```

### Downstream tasks
```bash
cd downstream/PointNet #  to evaluate with PointNet
python train_cls.py \
--log_dir path_to_results_folder \
--dataset_type modelnet40 \ # type of datase such as modelnet40, scanobjectnn
--dataset_path path_to_folder_dataset \
--model_path path_to_pre_trained_model \ # path to pre_trained models
```

## Contributions:

Following are the code wise contributions:

Our main contribution lies in the `pretrain_mod` directory. 
* Siddhant Shingi: `pretrain_mod/models/local_model.py` (defining the model architecture)
* Nidhi Chandra: `pretrain_mod/data_utils/ModelNetDataLoader.py` (defining the ModelNet40 dataloader - includes code for extracting small window for point-wise knowledge transfer)
* Arbaaz Qureshi: `pretrain_mod/train.py` (writing training loop for modified model architecture and dataloader)

All of us were involved in running the experiments (downstream tasks), brainstorming ideas and result analysis. 