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
        self.feature_dim = args.feature_dim
        self.num_views = args.num_views
        self.net2d = resnet50(weights="IMAGENET1K_V2")

        if args.model == "pointnet":
            self.net3d = PointNet_point_global(input_channels=3, feature_dim=args.feature_dim, feature_transform=True)
        elif args.model == "dgcnn":
            self.net3d = DGCNN_point_global(args)

    def forward(self, 
            point_cloud, 
            images, 
            local_images, 
            pix2point
        ):
        print (images.shape)
        feat_images = self.net2d(images)
        feat_local_images = self.net2d(local_images)
        print (feat_images.shape, feat_local_images.shape)

        kkkknsc
        pass
