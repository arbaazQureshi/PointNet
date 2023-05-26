import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50
from .pointnet_utils import PointNet_point_global
from .dgcnn_utils import DGCNN_point_global

class LocalModel(nn.Module):
    def __init__(self, 
            args,
        ):
        super(LocalModel, self).__init__()
        self.args = args
        self.net2d = resnet50(weights="IMAGENET1K_V2")
        self.net2d.fc = torch.nn.Linear(self.net2d.fc.in_features, 1024)
        self.g1 = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.args.feature_dim, bias=True),
        )
        self.g2 = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.args.feature_dim, bias=True),
        )

        if self.args.model == "pointnet":
            self.net3d = PointNet_point_global(input_channels=3, feature_dim=self.args.feature_dim, feature_transform=True)
        elif self.args.model == "dgcnn":
            self.net3d = DGCNN_point_global(self.args)

    def forward(self, 
            point_cloud, 
            images, 
            local_images, 
            num_pairs
        ):
        # images: B*V, C, H, W
        global_feat2d = self.net2d(images) # B*V, 1024
        global_feat2d = global_feat2d.view((int(global_feat2d.size(0) / self.args.num_views), self.args.num_views, -1)) # B, V, 1024
        global_feat2d = torch.max(global_feat2d, 1)[0] # B, 1024
        global_feat2d = self.g1(global_feat2d) # B, F

        # images: B*P, C, 16, 16
        pixel_feat2d = self.net2d(local_images) # B*P, 1024
        pixel_feat2d = pixel_feat2d.view((int(pixel_feat2d.size(0) / num_pairs), num_pairs, -1)) # B, P, 1024
        pixel_feat2d = self.g2(pixel_feat2d) # B, P, F

        T2 = None
        if self.args.model == "pointnet":
            global_feat3d, point_feat3d, T2 = self.net3d(point_cloud)
        elif self.args.model == "dgcnn":
            point_cloud = point_cloud.transpose(2, 1)
            global_feat3d, point_feat3d = self.net3d(point_cloud)

        return global_feat2d, pixel_feat2d, global_feat3d, point_feat3d, T2



python train_cls.py --model_path "../../pretrain_mod/ws_16/pre_trained_pointnet_epoch_80.pth" --log_dir "ws_16"
