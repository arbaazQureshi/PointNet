import argparse
import json
import os
import random

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from data_utils import ModelNetDataset, ModelNetDataset_H5PY, ScanObjectNNDataset
from models import PointNet, get_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import bn_momentum_adjust, copy_parameters, init_weights, init_zeros


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser("Classification")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size in training [default: 32]")
    parser.add_argument("--nepoch", default=250, type=int, help="number of epoch in training [default: 250]")
    parser.add_argument(
        "--learning_rate", default=0.001, type=float, help="learning rate in training [default: 0.001]"
    )
    parser.add_argument("--num_point", type=int, default=1024, help="Point Number [default: 1024]")
    parser.add_argument("--optimizer", type=str, default="Adam", help="optimizer for training [default: Adam]")
    parser.add_argument("--log_dir", type=str, default=None, help="experiment root")
    parser.add_argument("--model_path", type=str, default="", help="model pre-trained")
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")
    parser.add_argument("--dataset_type", type=str, required=True, help="scanobjectnn|modelnet40|scanobjectnn10")
    parser.add_argument("--lr_decay", type=float, default=0.7, help="decay rate for learning rate")
    parser.add_argument("--decay_step", type=int, default=20, help="decay step for ")
    parser.add_argument("--momentum_decay", type=float, default=0.5, help="momentum_decay decay of batchnorm")
    parser.add_argument("--manualSeed", type=int, default=None, help="random seed")
    parser.add_argument("--data_aug", action="store_true", help="Using data augmentation for training phase")
    parser.add_argument("--weight_decay", action="store_true", help="Using data augmentation for training phase")
    parser.add_argument("--ratio", type=int, default=1, metavar="S", help="random seed (default: None)")
    return parser.parse_args()


def get_global_embeddings():
    args = parse_args()
    print(args.manualSeed)
    if args.manualSeed is not None:
        random.seed(args.manualSeed)
        torch.manual_seed(args.manualSeed)
        np.random.seed(args.manualSeed)
    else:
        args.manualSeed = random.randint(1, 10000)  # fix seed
        print("Random Seed: ", args.manualSeed)
        random.seed(args.manualSeed)
        torch.manual_seed(args.manualSeed)
        np.random.seed(args.manualSeed)

    if args.dataset_type == "modelnet40":
        dataset = ModelNetDataset(
            root=args.dataset_path, npoints=args.num_point, split="train", data_augmentation=args.data_aug
        )
        test_dataset = ModelNetDataset(
            root=args.dataset_path, npoints=args.num_point, split="test", data_augmentation=False
        )
    elif args.dataset_type == "modelnet40h5py":
        dataset = ModelNetDataset_H5PY(
            filelist=args.dataset_path + "/train.txt", num_point=args.num_point, data_augmentation=args.data_aug
        )
        test_dataset = ModelNetDataset_H5PY(
            filelist=args.dataset_path + "/test.txt", num_point=args.num_point, data_augmentation=False
        )
    elif args.dataset_type == "scanobjectnn":
        dataset = ScanObjectNNDataset(
            root=args.dataset_path, npoints=args.num_point, split="train", data_augmentation=args.data_aug
        )

        test_dataset = ScanObjectNNDataset(
            root=args.dataset_path, split="test", npoints=args.num_point, data_augmentation=False
        )
    elif args.dataset_type == "scanobjectnnbg":
        dataset = ScanObjectNNDataset(
            root=args.dataset_path, npoints=args.num_point, split="train", data_augmentation=args.data_aug
        )

        test_dataset = ScanObjectNNDataset(
            root=args.dataset_path, split="test", npoints=args.num_point, data_augmentation=False
        )
    else:
        exit("wrong dataset type")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
    )

    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    print(len(dataset), len(test_dataset))
    num_classes = dataset.num_classes
    print("classes", num_classes)


    classifier = PointNet(3, num_classes, global_feature=True)
    classifier.apply(init_weights)
    classifier.stn1.mlp2[-1].apply(init_zeros)
    classifier.stn2.mlp2[-1].apply(init_zeros)
    
    if args.model_path != "":
        print("Copying parameters")
        classifier = copy_parameters(classifier, torch.load(args.model_path))
    
    classifier.cuda()
    
    if args.weight_decay:
        print("Using weight decay")
        optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
    else:
        print("None using weight decay")
        optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

    global_features = []
    all_targets = []

    with torch.no_grad():  # Test

        test_total_point = 0.0
        classifier.eval()
        
        for i, data in tqdm(enumerate(testdataloader, 0)):
            points, target = data
            test_total_point += points.size(0)
            target = target[:, 0]
            all_targets.append(target.data)
            points, target = points.cuda(), target.cuda()
            
            gf = classifier(points).data.cpu()

            global_features.append(gf)
            
    global_features = torch.cat(global_features, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    print(global_features.shape, all_targets.shape)
    
    os.makedirs(args.log_dir)
    np.save(args.log_dir + "/global_features.npy", global_features)
    np.save(args.log_dir + "/targets.npy", all_targets)
    
    


if __name__ == "__main__":
    get_global_embeddings()
