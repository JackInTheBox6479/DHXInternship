import torch
import torch.nn as nn

# (kernel_size, number of filters, stride, padding)
# "M" = maxpooling
# list is tuples followed by number of repeats

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self.create_conv_layers(self.architecture)
        self.fcs = self.create_fcs(**kwargs)

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if isinstance(x, tuple):
                layers += [
                    CNNBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3]),
                ]
                in_channels = x[1]

            elif isinstance(x, str):
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif isinstance(x, list):
                conv1 = x[0]
                conv2 = x[1]
                rpts = x[2]

                for n in range(rpts):
                    layers += [CNNBlock(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3])]
                    layers += [CNNBlock(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3])]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def create_fcs(self, split_size, num_boxes, num_classes):
        s, b, c = split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Linear(1024*s*s, 4096),
            nn.ReLU(),

            # S * S is grid boxes
            # B * 5 is [x, y, w, h, confidence]
            # C is class probabilities (number of classes you are reading for)
            # For 7x7 grid with three objects, final output is 7x7x13 (assuming two objects)
            nn.Linear(4096, s*s*(b*5+c))
        )