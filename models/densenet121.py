import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        inner_channels = 4 * growth_rate
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inner_channels, kernel_size=1, bias=False)
        self.bn2   = nn.BatchNorm2d(inner_channels)
        self.conv2 = nn.Conv2d(inner_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return torch.cat([x, out], 1)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn   = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseNet121(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5):
        super(DenseNet121, self).__init__()
        growth_rate       = 32
        block_config      = [6, 12, 24, 16]
        num_init_features = 2 * growth_rate

        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Dense Blocks + Transition Layers
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = self._make_dense_block(num_features, num_layers, growth_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = Transition(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        self.relu    = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)           # prevents overfitting on small datasets

        # Classification head
        self.classifier = nn.Linear(num_features, num_classes)

        # Weight initialization
        self._initialize_weights()

    def _make_dense_block(self, in_channels, num_layers, growth_rate):
        layers = []
        for i in range(num_layers):
            layers.append(Bottleneck(in_channels + i * growth_rate, growth_rate))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.features(x)
        out = self.relu(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)        # applied before classifier
        out = self.classifier(out)
        return out