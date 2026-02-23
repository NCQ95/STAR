import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict
import timm

class PseudoCombiner(nn.Module):
    def __init__(self, no_classes, pretrained=False, backbone_name="resnet18"):
        super(PseudoCombiner, self).__init__()
        self.backbone_name = backbone_name
        self.backbone, feature_dim = self.create_backbone(
            backbone_name, pretrained, no_classes
        )
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(feature_dim, no_classes)
        self.p_logvar = nn.Sequential(nn.Linear(512 * 1, 512),
                                      nn.ReLU())
        self.p_mu = nn.Sequential(nn.Linear(512 * 1, 512),
                                  nn.LeakyReLU())


    def forward(self, x=[], tg=[], train=False, flag=False):
        outputs = []
        pseudo_no = len(x)
        x = torch.cat(x, dim=0)
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = list(torch.split(x, int(x.shape[0] / pseudo_no), dim=0))

        if flag == True:
            z1 = x[0] 
            z2 = tg
            simm = torch.sum(z1 * z2, dim=1) / (torch.norm(z1, dim=1) * torch.norm(z2, dim=1))
            Lx = (simm ** 2).mean()
            end_points = {}
            t = torch.cat([z1, z2],dim=0)
            logvar = self.p_logvar(t)
            mu = self.p_mu(t)
            std = logvar.div(2).exp()
            eps = std.data.new(std.size()).normal_()
            mid = mu + 0.2 * std * eps
            end_points['logvar'] = logvar
            end_points['mu'] = mu
            end_points['Embedding'] = mid
            return end_points, Lx

        if train == True:
            end_points = {}
            t = torch.cat([x[0], x[1]], dim=0)
            logvar = self.p_logvar(t)
            mu = self.p_mu(t)
            std = logvar.div(2).exp()
            eps = std.data.new(std.size()).normal_()
            mid = mu + 0.2 * std * eps
            end_points['logvar'] = logvar
            end_points['mu'] = mu
            end_points['Embedding'] = mid
            t1 = mid[:len(x[0]), :]
            t2 = mid[len(x[0]):, :]
            x = [t1, t2]
            for idx, pseudo in enumerate(x):
                outputs.append(self.classifier(pseudo))
            return outputs, end_points

        if train == False:
            x.append(x[0])
            t = torch.cat([x[0], x[1]], dim=0)
            x = t[:len(x[0]), :]
            mu = self.p_mu(x)
            x = mu
            outputs.append(self.classifier(x))
            return outputs
            

    def create_backbone(self, backbone_name, pretrained, no_classes):

        if backbone_name == "resnet18":
            backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        elif backbone_name == "vit_small":
            backbone = vit_initialization(
                network_variant="vit_small_patch16_224", pretrained=pretrained
            )
            feature_dim = backbone.ft_dim
        elif backbone_name.lower() == "caffenet":
            backbone = AlexNetCaffe()
            for m in backbone.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, 0.1)
                    nn.init.constant_(m.bias, 0.0)
            if pretrained:
                state_dict = torch.load("./Pretrained_Models/alexnet_caffe.pth.tar")
                backbone.load_state_dict(state_dict, strict=False)
            backbone.classifier = backbone.classifier[:-1]
            feature_dim = 4096

        return backbone, feature_dim


class vit_initialization(nn.Module):
    def __init__(self, network_variant, pretrained):
        super(vit_initialization, self).__init__()

        self.model = timm.create_model(network_variant, pretrained=pretrained)
        self.ft_dim = self.model.head.in_features

    def forward(self, x):
        x = self.model.forward_features(x)
        x = x[:, 0]

        return x


class AlexNetCaffe(nn.Module):
    def __init__(self, dropout=True):
        super(AlexNetCaffe, self).__init__()
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
                    ("norm1", nn.LocalResponseNorm(5, 1.0e-4, 0.75)),
                    ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
                    ("norm2", nn.LocalResponseNorm(5, 1.0e-4, 0.75)),
                    ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
                    ("relu3", nn.ReLU(inplace=True)),
                    ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
                    ("relu4", nn.ReLU(inplace=True)),
                    ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
                    ("relu5", nn.ReLU(inplace=True)),
                    ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
                ]
            )
        )
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc6", nn.Linear(256 * 6 * 6, 4096)),
                    ("relu6", nn.ReLU(inplace=True)),
                    ("drop6", nn.Dropout()),
                    ("fc7", nn.Linear(4096, 4096)),
                    ("relu7", nn.ReLU(inplace=True)),
                    ("drop7", nn.Dropout()),
                    ("fc8", nn.Linear(4096, 1000)),
                ]
            )
        )

    def forward(self, x, train=True):
        x = self.features(x * 57.6)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class AugNet(nn.Module):
    def __init__(self, noise_lv):
        super(AugNet, self).__init__()
        self.noise_lv = nn.Parameter(torch.zeros(1))
        self.shift_var = nn.Parameter(torch.empty(3,216,216))
        nn.init.normal_(self.shift_var, 1, 0.1)
        self.shift_mean = nn.Parameter(torch.zeros(3, 216, 216))
        nn.init.normal_(self.shift_mean, 0, 0.1)

        self.shift_var2 = nn.Parameter(torch.empty(3, 212, 212))
        nn.init.normal_(self.shift_var2, 1, 0.1)
        self.shift_mean2 = nn.Parameter(torch.zeros(3, 212, 212))
        nn.init.normal_(self.shift_mean2, 0, 0.1)

        self.shift_var3 = nn.Parameter(torch.empty(3, 208, 208))
        nn.init.normal_(self.shift_var3, 1, 0.1)
        self.shift_mean3 = nn.Parameter(torch.zeros(3, 208, 208))
        nn.init.normal_(self.shift_mean3, 0, 0.1)

        self.shift_var4 = nn.Parameter(torch.empty(3, 220, 220))
        nn.init.normal_(self.shift_var4, 1, 0.1)
        self.shift_mean4 = nn.Parameter(torch.zeros(3, 220, 220))
        nn.init.normal_(self.shift_mean4, 0, 0.1)

        self.Device = torch.device("cuda:0")
        self.norm = nn.InstanceNorm2d(3)

        self.spatial = nn.Conv2d(3, 3, 9).to(self.Device)
        self.spatial_up = nn.ConvTranspose2d(3, 3, 9).to(self.Device)

        self.spatial2 = nn.Conv2d(3, 3, 13).to(self.Device)
        self.spatial_up2 = nn.ConvTranspose2d(3, 3, 13).to(self.Device)

        self.spatial3 = nn.Conv2d(3, 3, 17).to(self.Device)
        self.spatial_up3 = nn.ConvTranspose2d(3, 3, 17).to(self.Device)

        self.spatial4 = nn.Conv2d(3, 3, 5).to(self.Device)
        self.spatial_up4 = nn.ConvTranspose2d(3, 3, 5).to(self.Device)
        self.color = nn.Conv2d(3, 3, 1).to(self.Device)

        for param in list(list(self.color.parameters()) +
                          list(self.spatial.parameters()) + list(self.spatial_up.parameters()) +
                          list(self.spatial2.parameters()) + list(self.spatial_up2.parameters()) +
                          list(self.spatial3.parameters()) + list(self.spatial_up3.parameters()) +
                          list(self.spatial4.parameters()) + list(self.spatial_up4.parameters())
                          ):
            param.requires_grad=False

    def forward(self, x, estimation=False):
        if not estimation:
            spatial = nn.Conv2d(3, 3, 9).to(self.Device)
            spatial_up = nn.ConvTranspose2d(3, 3, 9).to(self.Device)

            spatial2 = nn.Conv2d(3, 3, 13).to(self.Device)
            spatial_up2 = nn.ConvTranspose2d(3, 3, 13).to(self.Device)

            spatial3 = nn.Conv2d(3, 3, 17).to(self.Device)
            spatial_up3 = nn.ConvTranspose2d(3, 3, 17).to(self.Device)

            spatial4 = nn.Conv2d(3, 3, 5).to(self.Device)
            spatial_up4 = nn.ConvTranspose2d(3, 3, 5).to(self.Device)

            color = nn.Conv2d(3,3,1).to(self.Device)
            weight = torch.randn(5)

            x = x + torch.randn_like(x) * self.noise_lv * 0.01
            x_c = torch.tanh(F.dropout(color(x), p=.2))

            x_sdown = spatial(x)
            x_sdown = self.shift_var * self.norm(x_sdown) + self.shift_mean
            x_s = torch.tanh(spatial_up(x_sdown))
            
            x_s2down = spatial2(x)
            x_s2down = self.shift_var2 * self.norm(x_s2down) + self.shift_mean2
            x_s2 = torch.tanh(spatial_up2(x_s2down))

            x_s3down = spatial3(x)
            x_s3down = self.shift_var3 * self.norm(x_s3down) + self.shift_mean3
            x_s3 = torch.tanh(spatial_up3(x_s3down))

            x_s4down = spatial4(x)
            x_s4down = self.shift_var4 * self.norm(x_s4down) + self.shift_mean4
            x_s4 = torch.tanh(spatial_up4(x_s4down))
            output = (weight[0] * x_c + weight[1] * x_s + weight[2] * x_s2+ weight[3] * x_s3 + weight[4]*x_s4) / weight.sum()
        else:
            x = x + torch.randn_like(x) * self.noise_lv * 0.01
            x_c = torch.tanh(self.color(x))
            
            x_sdown = self.spatial(x)
            x_sdown = self.shift_var * self.norm(x_sdown) + self.shift_mean
            x_s = torch.tanh(self.spatial_up(x_sdown))
            
            x_s2down = self.spatial2(x)
            x_s2down = self.shift_var2 * self.norm(x_s2down) + self.shift_mean2
            x_s2 = torch.tanh(self.spatial_up2(x_s2down))

            x_s3down = self.spatial3(x)
            x_s3down = self.shift_var3 * self.norm(x_s3down) + self.shift_mean3
            x_s3 = torch.tanh(self.spatial_up3(x_s3down))

            x_s4down = self.spatial4(x)
            x_s4down = self.shift_var4 * self.norm(x_s4down) + self.shift_mean4
            x_s4 = torch.tanh(self.spatial_up4(x_s4down))
            output = (x_c + x_s + x_s2 + x_s3 + x_s4) / 5
        return output