import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet


class ResNetFace(nn.Module):
    def __init__(self, name, emb_dim):
        super().__init__()
        available_models = {
                'resnet18': ['basic', [2, 2, 2, 2]],
                'resnet34': ['basic', [3, 4, 6, 3]],
                'resnet50': ['bottleneck', [3, 4, 6, 3]],
                'resnet101': ['bottleneck', [3, 4, 23, 3]],
                'resnet152': ['bottleneck', [3, 8, 36, 3]],
                'sphereface4': [0, 0, 0, 0],
                'sphereface10': [0, 1, 2, 0],
                'sphereface20': [1, 2, 4, 1],
                'sphereface36': [2, 4, 8, 2],
                'sphereface64': [3, 8, 16, 3],
                'uniformface': [2, 3, 5, 2]  # from Figure 3 of UniformFace paper
                }
        if name not in available_models.keys():
            raise ValueError(f'Invalid model name. Got: {name}')

        if name[0] == 's':
            self.model = SphereFace(available_models[name], emb_dim)
        elif name[0] == 'u':
            self.model = UniformFace(available_models[name], emb_dim)
        else:
            block_name = available_models[name][0]
            block = BasicBlock if block_name == 'basic' else Bottleneck
            self.model = ResNet(block, available_models[name][1], emb_dim)

    def forward(self, x):
        return self.model(x)


class SphereBaiscBlock(nn.Module):
    def __init__(self, inplanes, planes, num_residual_layer, pooling=False):
        super().__init__()
        # pooling argument is used for UniformFace
        self.pooling = pooling
        if pooling:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Each block has at least one Conv2d with stride 2 (sphereface paper Table. 2)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.prelu = nn.PReLU(planes)
        residual_layers = []
        for _ in range(num_residual_layer):
            inner_conv = [
                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
                nn.PReLU(planes),
                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
                nn.PReLU(planes)
            ]
            residual_layers.append(nn.Sequential(*inner_conv))
        self.conv2 = nn.ModuleList(residual_layers)

    def forward(self, x):
        x = self.prelu(self.conv1(x))
        if self.pooling:
            x = self.maxpool(x)
        for m in self.conv2:
            output = m(x)
            x = output + x
        return x


class SphereFace(nn.Module):
    def __init__(self, num_residual_layers, emb_dim):
        super().__init__()
        if len(num_residual_layers) != 4:
            # SphereFace is composed of 4 convolutinal blocks
            raise ValueError('Invalid number of layers for SphereFace. len(num_residual_layers) shoud be 4')
        self.num_residual_layers = num_residual_layers
        layers = []
        planes = [3, 64, 128, 256, 512]
        for i, n_layer in enumerate(num_residual_layers):
            layers.append(SphereBaiscBlock(planes[i], planes[i+1], n_layer))
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512*7*7, emb_dim)

    def forward(self, x):
        out = self.layers(x)
        out = self.fc(out.view(x.size(0), -1))
        return out


class UniformFace(nn.Module):
    def __init__(self, num_residual_layers, emb_dim):
        super().__init__()
        layers = []
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU(64)
        planes = [64, 64, 128, 256, 512]
        for i, n_layer in enumerate(num_residual_layers):
            layers.append(SphereBaiscBlock(planes[i], planes[i+1], n_layer, True))
        self.layers = nn.Sequential(*layers)
        #self.fc = nn.Linear(512*7*7, emb_dim)

    def forward(self, x):
        out = self.prelu(self.conv1(x))
        out = self.layers(out)
        return out.view(x.size(0), -1)
