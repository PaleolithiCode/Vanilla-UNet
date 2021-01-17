# By Jim Chen / 2021/01/27

import PIL.Image as Image
import numpy as np
import os
import cv2

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torchvision.transforms import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])

y_transforms = transforms.ToTensor()

def diceCoeff(pred, gt, eps = 1e-5):
 
    # activation_func = nn.Tanh()
    activation_func = nn.Sigmoid()

    pred = activation_func(pred)

    n = gt.size(0)
    pred_flat = pred.view(n, -1)
    gt_flat = gt.view(n, -1)

    tp = torch.sum(gt_flat * pred_flat, dim = 1)
    fp = torch.sum(pred_flat, dim = 1) - tp
    fn = torch.sum(gt_flat, dim = 1) - tp

    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)

    return loss.sum() / n

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

def train_model(model, criterion, optimizer, train_dataload, num_epochs = 20):
    global device

    for epoch in range(num_epochs):

        print(
            'Epoch\t{} / {}\n'.format(epoch + 1, num_epochs) +
            '------------------------------------------'
            )

        data_size = len(train_dataload.dataset)
        epoch_loss = 0
        step = 0

        for img_batch, mask_batch in train_dataload:
            step += 1

            imgs = Variable(img_batch.to(device), requires_grad = True)
            labels = Variable(mask_batch.to(device), requires_grad = True)

            optimizer.zero_grad()

            outputs = model(imgs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            print("\t{}/{}\t\ttrain_loss : {:.3f}".format(step, (data_size - 1) // train_dataload.batch_size + 1, loss.item()))

        print(
            '------------------------------------------\n' +
            "epoch {}\tloss : {:.3f}\n".format(epoch + 1, epoch_loss / step) + 
            '------------------------------------------'
            )

    return model

def train_eval_model(model, criterion, optimizer, train_dataload, val_dataload, log_name, num_epochs = 20):
    global device

    for epoch in range(num_epochs):

        print(
            'Epoch\t{} / {}\n'.format(epoch + 1, num_epochs) +
            '------------------------------------------'
            )

        data_size = len(train_dataload.dataset)
        epoch_loss = 0
        step = 0

        for img_batch, mask_batch in train_dataload:
            step += 1

            imgs = Variable(img_batch.to(device), requires_grad = True)
            masks = Variable(mask_batch.to(device), requires_grad = True)

            optimizer.zero_grad()

            outputs = model(imgs)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            print("\t{}/{}\t\ttrain_loss : {:.3f}".format(step, (data_size - 1) // train_dataload.batch_size + 1, loss.item()))

        with torch.no_grad():

            val_acc = 0
            val_step = 0

            for img, mask in val_dataload:
                val_step += 1

                predict = model(img.to(device)).sigmoid()

                img_gt = torch.Tensor.cpu(torch.squeeze(mask)).numpy()

                img_predict = torch.Tensor.cpu(torch.squeeze(predict)).numpy()
                img_predict = img_predict > 0.5

                img_gt = np.array(img_gt, dtype = bool)
                img_predict = np.array(img_predict, dtype = bool)
                dice = np.sum(img_predict[img_gt == 1])*2.0 / (np.sum(img_predict) + np.sum(img_gt))

                val_acc += dice

        print(
            '------------------------------------------\n' +
            "epoch {}\tloss : {:.3f}\n".format(epoch + 1, epoch_loss / step) + 
            "epoch {}\tacc : {:.3f}\n".format(epoch + 1, val_acc / len(val_dataload)) + 
            '------------------------------------------'
            )

        with open('./logs/{}'.format(log_name), 'a', encoding = 'utf-8') as fileOut:
            fileOut.write("epoch {}\tloss : {:.3f}\n".format(epoch + 1, epoch_loss / step))
            fileOut.write("epoch {}\tacc : {:.3f}\n".format(epoch + 1, val_acc / len(val_dataload)))

    return model

def train(batch_size = 3):
    model = Unet(3, 1).to(device)

    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters())

    ct_dataset = CarpalTunnelDataSet("carpalTunnel", 'CT', transform = x_transforms, mask_transform = y_transforms)
    ct_dataloader = DataLoader(ct_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

    train_model(model, criterion, optimizer, ct_dataloader, 5)

class UnetCt():
    def __init__(self, root = "carpalTunnel", target = "CT", batch_size = 3, epoches = 5, criterion = 'DiceLoss', optimizer = 'Adam'):
        global device
        self.model = Unet(3, 1).to(device)
        self.root = './dataset/' + root
        self.target = target
        self.epoch = epoches
        self.batch_size = batch_size
        self.transform = x_transforms
        self.mask_transform = y_transforms

        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = DiceLoss()
        # self.criterion = nn.BCEWithLogitsLoss()

    def run(self):
        self.dataset = CarpalTunnelDataSet(self.root, self.target, transform = x_transforms, mask_transform = y_transforms)
        self.data_loader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True, num_workers = 4)
        train_model(self.model, self.criterion, self.optimizer, self.data_loader, self.epoch)

        torch.save(self.model.state_dict(), './weights_b{}_{}_{}.pth'.format(self.batch_size, self.target, self.epoch))

    def cross_val_run(self, i = 0):
        if '_' not in self.root:
            print("ONLY AVAILABLE WHEN ROOT == 'carpalTunnel_'")
            return

        self.train_dataset = CarpalTunnelDataSet(os.path.join(self.root, 'train_{}'.format(i + 1)), self.target, transform = x_transforms, mask_transform = y_transforms)
        self.train_loader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True, num_workers = 4)

        self.val_dataset = CarpalTunnelDataSet(os.path.join(self.root, 'val_{}'.format(i + 1)), self.target, transform = x_transforms, mask_transform = y_transforms)
        self.val_loader = DataLoader(self.val_dataset, batch_size = 1)

        train_eval_model(self.model, self.criterion, self.optimizer, self.train_loader, self.val_loader, '{}_{}_cross_val_{}.logs'.format(self.target, self.epoch, i + 1), self.epoch)

        torch.save(self.model.state_dict(), './cross_val/weights_{}_{}_cross_val_{}.pth'.format(self.target, self.epoch, i + 1))

    def load(self, path = './weights_5.pth'):
        global device
        self.model.load_state_dict(torch.load(path, map_location = device))

    def _optimizer_to(self, device):
        for param in self.optimizer.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

    def wipe_memory(self):
        global device
        self._optimizer_to(device)
        del self.optimizer
        del self.model
        torch.cuda.empty_cache()

    def selected_eval(self, x_path, y_path):
        img_x = Image.open(x_path).convert('RGB')
        img_y = Image.open(y_path)

        img_x = self.transform(img_x)
        img_y = self.mask_transform(img_y)

        data_loader = DataLoader([(img_x[:3], img_y)], batch_size = 1)

        with torch.no_grad():
            for img, mask in data_loader:
                predict = self.model(img.to(device)).sigmoid()
                img_predict = torch.Tensor.cpu(torch.squeeze(predict)).numpy()

                img_ori = cv2.imread(x_path)
                img_gt = cv2.imread(y_path)

                cv2.imshow('ori', img_ori)
                cv2.imshow('gt', img_gt)
                cv2.imshow('pred', img_predict)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def eval(self):
        global device
        self.dataset = CarpalTunnelDataSet(self.root, self.target, transform = x_transforms, mask_transform = y_transforms)
        self.data_loader = DataLoader(self.dataset, batch_size = 1)

        self.model.eval()

        cv2.namedWindow('ori')
        cv2.namedWindow('gt')
        cv2.namedWindow('pred')

        with torch.no_grad():
            for img, mask in self.data_loader:
                predict = self.model(img.to(device)).sigmoid()

                tmp = torch.Tensor.cpu(img).numpy()
                
                img_ori = np.zeros((tmp[0].shape[1], tmp[0].shape[2], 3))

                img_ori[:, :, 0] = tmp[0, 0, :, :]
                img_ori[:, :, 1] = tmp[0, 1, :, :]
                img_ori[:, :, 2] = tmp[0, 2, :, :]

                img_gt = torch.Tensor.cpu(torch.squeeze(mask)).numpy()
                img_predict = torch.Tensor.cpu(torch.squeeze(predict)).numpy()

                cv2.imshow('ori', img_ori)
                cv2.imshow('gt', img_gt)
                cv2.imshow('pred', img_predict)

                np_gt = np.array(img_gt, dtype = bool)
                np_predict = np.array(img_predict > 0.5, dtype = bool)
                dice = np.sum(np_predict[np_gt == 1])*2.0 / (np.sum(np_predict) + np.sum(np_gt))
                print("DC :", dice)

                cv2.waitKey(0)
                cv2.destroyAllWindows()

                break

if __name__ == "__main__":
    # torch.cuda.empty_cache()

    # def fold_val(unet_list):
    #     for i, model in enumerate(unet_list):
    #         model.cross_val_run(i)
    #         model.wipe_memory()

    # mn_unet = [UnetCt(root = 'carpalTunnel_', target = 'MN', batch_size = b_size, epoches = epoch) for i in range(5)]
    # fold_val(mn_unet)
    # del mn_unet

    # ft_unet = [UnetCt(root = 'carpalTunnel_', target = 'FT', batch_size = b_size, epoches = epoch) for i in range(5)]
    # fold_val(ft_unet)
    # del ft_unet

    # ct_unet = [UnetCt(root = 'carpalTunnel_', target = 'CT', batch_size = b_size, epoches = epoch) for i in range(5)]
    # fold_val(ct_unet)
    # del ct_unet

    epoch = 20
    b_size = 3

    mn_unet = UnetCt(root = 'carpalTunnel', target = 'MN', batch_size = b_size, epoches = epoch)
    ft_unet = UnetCt(root = 'carpalTunnel', target = 'FT', batch_size = b_size, epoches = epoch)
    ct_unet = UnetCt(root = 'carpalTunnel', target = 'CT', batch_size = b_size, epoches = epoch)

    mn_unet.run()
    mn_unet.wipe_memory()

    ft_unet.run()
    ft_unet.wipe_memory()

    ct_unet.run()
    ct_unet.wipe_memory()