import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import PIL.Image as Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])

y_transforms = transforms.ToTensor()

def diceCoeff(pred, gt, eps = 1e-5):

    activation_fn = nn.Sigmoid()
    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim = 1)
    fp = torch.sum(pred_flat, dim = 1) - tp
    fn = torch.sum(gt_flat, dim = 1) - tp

    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)

    return loss.sum() / N

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def	forward(self, input, target):
        return 1- diceCoeff(input, target)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_ch, out_ch, 3, padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True),

            nn.Conv2d(out_ch, out_ch, 3, padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True)

        )

    def forward(self, input):
        return self.conv(input)

class InputConvolution(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InputConvolution, self).__init__()
        self.inp_conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.inp_conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size = 2, stride = 2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        x = torch.cat([x2, x1], dim = 1)
        x = self.conv(x)

        return x

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class LastConvolution(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LastConvolution, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv1(x)
        return x

class Unet(nn.Module):
    def __init__(self, channels, classes):
        super(Unet, self).__init__()
        self.inp = InputConvolution(channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.out = LastConvolution(64, classes)

    def forward(self, x):
        x1 = self.inp(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        x10 = self.out(x9)
        return x10

def make_dataset(root = 'carpalTunnel', mask = 'CT'):
    img_list = []
    src_dir = ['T1', 'T2']

    for dir_ in os.listdir(root):
        sub_dir = os.path.join(root, dir_)
        mask_dir = os.path.join(sub_dir, mask)

        data_len = len(os.listdir(mask_dir))

        for i in range(data_len):
            for src in src_dir:
                src_img = os.path.join(sub_dir, src, '{}.jpg'.format(i))
                mask_img = os.path.join(mask_dir, '{}.jpg'.format(i))
                img_list.append((src_img, mask_img))

    return img_list

class CarpalTunnelDataSet(Dataset):
    def __init__(self, root, mask, transform = None, mask_transform = None):
        dataset = make_dataset(root, mask)
        self.dataset = dataset

        self.transform = transform
        self.mask_transform = mask_transform

    def __getitem__(self, index):
        x_path, y_path = self.dataset[index]

        img_x = Image.open(x_path).convert('RGB')
        img_y = Image.open(y_path)

        if self.transform != None:
            img_x = self.transform(img_x)

        if self.mask_transform != None:
            img_y = self.mask_transform(img_y)

        return img_x[ : 3], img_y[ : 3]

    def __len__(self):
        return len(self.dataset)
