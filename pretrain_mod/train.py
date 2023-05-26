import argparse
import json
import os

import torch
import torch.nn.functional as F
from data_utils import ModelNetSSLDataset
from models import DGCNN_point_global, Model2D_pixel, PointNet_point_global, feature_transform_regularizer
from models import LocalModel
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import contrastive_loss
import time


def parse_args():
    
    """PARAMETERS"""
    
    parser = argparse.ArgumentParser("self supervised parameters")
    
    parser.add_argument("--feature_dim", default=128, type=int, help="Feature dim for latent vector")
    parser.add_argument("--temperature", default=0.5, type=float, help="Temperature used in softmax")
    parser.add_argument("--batch_size", default=32, type=int, help="Number of images in each mini-batch")
    parser.add_argument("--nepoch", default=100, type=int, help="Number of sweeps over the dataset to train")
    parser.add_argument("--emb_dims", type=int, default=1024, metavar="N", help="Dimension of embeddings")
    parser.add_argument("--use_sgd", action="store_true", help="Use SGD")
    parser.add_argument("--fps", type=str, default="random", help="sampling pair, fps, sps or random")
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001, 0.1 if using sgd)")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="SGD momentum (default: 0.9)")
    parser.add_argument("--scheduler", type=str, default="cos", metavar="N", choices=["cos", "step", "None"], help="Scheduler to use, [cos, step]")
    
    parser.add_argument("--model", type=str, default="pointnet", help="model pre-training [pointnet or dgcnn]")
    parser.add_argument("--num_views", default=12, type=int, help="number of views image")
    parser.add_argument("--num_point_contrast", default=512, type=int, help="number of views image")
    parser.add_argument("--num_points", default=1024, type=int, help="number points")
    parser.add_argument("--k", type=int, default=20, metavar="M", help="number nearest neighboor")
    parser.add_argument("--window_size", type=int, default=16, help="window size")
    parser.add_argument("--resnet_unfreeze", action="store_true", help="unfreeze resnet")
    
    parser.add_argument("--dataset", type=str, default="/work/nidhichandra_umass_edu/data/ModelNet40_blender_sampling_1024/ModelNet40_blender_sampling_1024", help="Path dataset")
    parser.add_argument("--pre_fix", type=str, default="ModelNet40_MV", help="name folder multi-views")
    parser.add_argument("--log_dir", type=str, default="ws_8", help="folder results")
    parser.add_argument("--path_model", type=str, default="results_ssl_full32/128_0.5_512_1000_model_epoch_1000.pth", help="path model 2d")
    
    return parser.parse_args()


if __name__ == "__main__":

    # args parse
    args = parse_args()
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    with open("%s/args.txt" % args.log_dir, "w") as f:
        json.dump(args.__dict__, f, indent=2)
    print ('loading data...')
    # load data
    train_data = ModelNetSSLDataset(
        root=args.dataset,
        npoints=args.num_points,
        data_augmentation=False,
        num_views=args.num_views,
        num_point_contrast=args.num_point_contrast,
        fps=args.fps,
        pre_fix=args.pre_fix,
        window_size=args.window_size,
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    print(args)
    print ('loading model...')
    # define model
    model = torch.nn.DataParallel(LocalModel(args))
    model.cuda()
    # freeze feature extractor in 2d networks
    if not args.resnet_unfreeze:
        for param in model.module.net2d.parameters():
            param.requires_grad = False
        for param in model.module.net2d.fc.parameters():
            param.requires_grad = True

    print ('loading optimizer, scheduler...')
    parameter_list = list(model.module.net2d.fc.parameters()) + list(model.module.net3d.parameters()) + list(model.module.g1.parameters()) + list(model.module.g2.parameters())
    if args.use_sgd:
        print("Use SGD")
        optimizer = optim.SGD(
            parameter_list,
            lr=args.lr * 100,
            momentum=args.momentum,
            weight_decay=1e-4,
        )
    else:
        print("Use Adam")
        optimizer = optim.Adam(parameter_list, lr=args.lr)
    if args.scheduler == "cos":
        scheduler = CosineAnnealingLR(optimizer, args.nepoch, eta_min=1e-3)
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer, 20, 0.7)
    print ('model training...')
    model.train()
    for epoch in range(0, args.nepoch):
        total_loss = 0.0
        total_seen = 0.0
        total_loss_point = 0.0
        total_loss_global = 0.0
        for data in tqdm(train_loader, total=len(train_loader)):
            point_cloud, images, local_images, pix2point = data
            B, V, C, H, W = images.size()
            images = images.view(-1, C, H, W)
            _, num_pairs, local_C, local_H, local_W = local_images.size()
            local_images = local_images.view(-1, local_C, local_H, local_W)
            for m in range(B):
                pix2point[m][:, 0] += m * V
            # pix2point = pix2point
            point_cloud, images, local_images = point_cloud.cuda(), images.cuda(), local_images.cuda()

            total_seen += B
            optimizer.zero_grad()
            
            global_feat2d, pixel_feat2d, global_feat3d, point_feat3d, T2 = model(point_cloud, images, local_images, num_pairs)
            loss_reg = 0.0
            if args.model == "pointnet":
                loss_reg = feature_transform_regularizer(T2) * 0.001

            loss_global = F.mse_loss(global_feat3d, global_feat2d)
            loss_point = 0.0
            for id in range(B):
                loss_point += contrastive_loss(
                    F.normalize(point_feat3d[id][:, pix2point[id][:, 3]].transpose(1, 0), dim=-1),
                    F.normalize(
                        pixel_feat2d[id], dim=-1
                    ),
                    args.temperature,
                )
            loss_point /= B

            loss = loss_point + 100 * loss_global + loss_reg
            total_loss_global += loss_global.item() * B
            total_loss_point += loss_point.item() * B
            total_loss += loss.item() * B
            loss.backward()
            optimizer.step()
        if args.scheduler == "cos":
            scheduler.step()
        elif args.scheduler == "step":
            if optimizer.param_groups[0]["lr"] > 1e-5:
                scheduler.step()
            if optimizer.param_groups[0]["lr"] < 1e-5:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 1e-5
        print(
            "Epoch [%s]: loss: %s, loss point: %s, loss global: %s"
            % (epoch, total_loss / total_seen, total_loss_point / total_seen, total_loss_global / total_seen)
        )
        if epoch % 20 == 0:
            print ('saving model')
            if args.model == "pointnet":
                torch.save(model.module.net3d.state_dict(), "%s/pre_trained_pointnet_epoch_%d.pth" % (args.log_dir, epoch))
            elif args.model == "dgcnn":
                torch.save(model.module.net3d.state_dict(), "%s/pre_trained_dgcnn_epoch_%d.pth" % (args.log_dir, epoch))
    if args.model == "pointnet":
        torch.save(net3d.state_dict(), "%s/pre_trained_pointnet.pth" % (args.log_dir))
    elif args.model == "dgcnn":
        torch.save(net3d.state_dict(), "%s/pre_trained_dgcnn.pth" % (args.log_dir))
