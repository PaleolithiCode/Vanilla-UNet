import sys
import os
import cv2
import torch

import PIL.Image as Image
import numpy as np

from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from structure.unet import Unet, CarpalTunnelDataSet, DiceLoss
from ui.mainwindow import Ui_CarpalTunnelUNetPrediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])

y_transforms = transforms.ToTensor()

def dc_compute(img_gt, img_predict):
    np_gt = np.array(img_gt, dtype = bool)
    np_predict = np.array(img_predict > 0.5, dtype = bool)
    dice = np.sum(np_predict[np_gt == 1]) * 2.0 / (np.sum(np_predict) + np.sum(np_gt))

    return dice

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

    def load(self, path = './weights_5.pth'):
        global device
        self.model.load_state_dict(torch.load(path, map_location = device))

    def selected_eval(self, x_path, y_path):
        img_x = Image.open(x_path).convert('RGB')
        img_y = Image.open(y_path)

        img_x = self.transform(img_x)
        img_y = self.mask_transform(img_y)

        data_loader = DataLoader([(img_x[:3], img_y)], batch_size = 1)

        with torch.no_grad():
            for img, mask in data_loader:
                predict = self.model(img.to(device)).sigmoid()
                img_gt = torch.Tensor.cpu(torch.squeeze(mask)).numpy()
                img_predict = torch.Tensor.cpu(torch.squeeze(predict)).numpy()

                dice = dc_compute(img_gt, img_predict)
                
                return dice, img_predict

def read_log(pid):
    with open('./eval_logs/p{}_MN_b3_e20.logs'.format(pid), 'r', encoding = 'utf-8') as fileIn:
        mn_mean = fileIn.readlines()[0].strip().split()[-1]

    with open('./eval_logs/p{}_FT_b3_e20.logs'.format(pid), 'r', encoding = 'utf-8') as fileIn:
        ft_mean = fileIn.readlines()[0].strip().split()[-1]

    with open('./eval_logs/p{}_CT_b3_e20.logs'.format(pid), 'r', encoding = 'utf-8') as fileIn:
        ct_mean = fileIn.readlines()[0].strip().split()[-1]

    return mn_mean, ft_mean, ct_mean

class ui_window(QtWidgets.QMainWindow):

    def __init__(self):
        super(ui_window, self).__init__()
        self.ui = Ui_CarpalTunnelUNetPrediction()
        self.ui.setupUi(self)
        self.mn_unet = UnetCt(root = 'carpalTunnel', target = 'MN')
        self.ft_unet = UnetCt(root = 'carpalTunnel', target = 'FT')
        self.ct_unet = UnetCt(root = 'carpalTunnel', target = 'CT')

        self.mn_unet.load('./models/bsize=3/weights_MN_20.pth')
        self.ft_unet.load('./models/bsize=3/weights_FT_20.pth')
        self.ct_unet.load('./models/bsize=3/weights_CT_20.pth')

        self.pid_selected = False
        self.__button_listener__()
        self.__slider_listener__()

    def __next__(self, first_trig = False):
        if not self.pid_selected:
            return

        if not first_trig:
            if self.ui.slider_patient_id.value() != self.sequence_length - 1:
                self.ui.slider_patient_id.setValue(self.ui.slider_patient_id.value() + 1)
            else:
                self.ui.slider_patient_id.setValue(0)

        img_id = self.ui.slider_patient_id.value()

        self.ui.lb_slider.setText("{:02d}/{}".format(self.ui.slider_patient_id.value(), self.sequence_length - 1))

        t1_path = './dataset/carpalTunnel/{}/{}/{}.jpg'.format(self.patient_id, 'T1', img_id)
        t2_path = './dataset/carpalTunnel/{}/{}/{}.jpg'.format(self.patient_id, 'T2', img_id)
        mn_path = './dataset/carpalTunnel/{}/{}/{}.jpg'.format(self.patient_id, 'MN', img_id)
        ft_path = './dataset/carpalTunnel/{}/{}/{}.jpg'.format(self.patient_id, 'FT', img_id)
        ct_path = './dataset/carpalTunnel/{}/{}/{}.jpg'.format(self.patient_id, 'CT', img_id)

        mn_pred = [self.mn_unet.selected_eval(t1_path, mn_path), self.mn_unet.selected_eval(t2_path, mn_path)]
        ft_pred = [self.ft_unet.selected_eval(t1_path, ft_path), self.ft_unet.selected_eval(t2_path, ft_path)]
        ct_pred = [self.ct_unet.selected_eval(t1_path, ct_path), self.ct_unet.selected_eval(t2_path, ct_path)]

        def cvt2qpix_and_edge_detect(npimg):
            img = Image.fromarray(np.uint8((npimg > 0.5) * 255))
            np_img = np.array(img)

            edge = cv2.Canny(np_img, 100, 200)

            return img.toqpixmap(), edge

        def edge_drawing(img, edge, colour = 'R'):
            counts = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

            img_out = img.copy()

            if colour == 'R':
                colouring = (255, 0, 0)
            elif colour == 'B':
                colouring = (0, 0, 255)
            else:
                colouring = (255, 255, 0)

            cv2.drawContours(img_out, counts, -1, colouring, 2)

            return img_out

        mn_pred.sort(reverse = True)
        mn_pred_dc = mn_pred[0][0]
        mn_pred, mn_edge = cvt2qpix_and_edge_detect(mn_pred[0][1])

        ft_pred.sort(reverse = True)
        ft_pred_dc = ft_pred[0][0]
        ft_pred, ft_edge = cvt2qpix_and_edge_detect(ft_pred[0][1])

        ct_pred.sort(reverse = True)
        ct_pred_dc = ct_pred[0][0]
        ct_pred, ct_edge = cvt2qpix_and_edge_detect(ct_pred[0][1])
        
        img = cv2.imread(t1_path, 1)
        img = edge_drawing(img, mn_edge, 'Y')
        img = edge_drawing(img, ft_edge, 'B')
        img = edge_drawing(img, ct_edge, 'R')

        sample_size = QSize(224, 224)
        mask_size = QSize(160, 160)
        result_size = QSize(320, 320)

        self.ui.lb_img_t1.setPixmap(QPixmap(t1_path).scaled(sample_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.ui.lb_img_t2.setPixmap(QPixmap(t2_path).scaled(sample_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.ui.lb_img_mn_gt.setPixmap(QPixmap(mn_path).scaled(mask_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.ui.lb_img_ft_gt.setPixmap(QPixmap(ft_path).scaled(mask_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.ui.lb_img_ct_gt.setPixmap(QPixmap(ct_path).scaled(mask_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.ui.lb_img_mn_pred.setPixmap(QPixmap(mn_pred).scaled(mask_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.ui.lb_img_ft_pred.setPixmap(QPixmap(ft_pred).scaled(mask_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.ui.lb_img_ct_pred.setPixmap(QPixmap(ct_pred).scaled(mask_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.ui.lb_single_mn_val.setText('{:.3f}'.format(mn_pred_dc))
        self.ui.lb_single_ft_val.setText('{:.3f}'.format(ft_pred_dc))
        self.ui.lb_single_ct_val.setText('{:.3f}'.format(ct_pred_dc))

        self.ui.lb_img_result.setPixmap(Image.fromarray(img).toqpixmap().scaled(result_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def __select__(self):
        self.__next__(True)

    def __slide__(self):
        if self.pid_selected:
            self.ui.lb_slider.setText("{:02d}/{}".format(self.ui.slider_patient_id.value(), self.sequence_length - 1))

    def __slider_listener__(self):
        self.ui.slider_patient_id.valueChanged.connect(self.__slide__)

    def __patient_select__(self):
        self.patient_id = int(str(self.ui.cBox_pid.currentText()))
        self.pid_selected = True

        self.sequence_length = len(os.listdir('./dataset/carpalTunnel/{}/T1'.format(self.patient_id)))

        mn_mean, ft_mean, ct_mean = read_log(self.patient_id)

        self.ui.lb_seq_mn_val.setText('{}'.format(mn_mean))
        self.ui.lb_seq_ft_val.setText('{}'.format(ft_mean))
        self.ui.lb_seq_ct_val.setText('{}'.format(ct_mean))

        self.ui.slider_patient_id.setMinimum(0)
        self.ui.slider_patient_id.setMaximum(self.sequence_length - 1)

        self.ui.slider_patient_id.setValue(0)
        self.ui.lb_slider.setText("{:02d}/{}".format(self.ui.slider_patient_id.value(), self.sequence_length - 1))
        self.__next__(True)

    def __button_listener__(self):
        self.ui.b_pidConfirm.clicked.connect(self.__patient_select__)
        self.ui.b_sliderNext.clicked.connect(self.__next__)
        self.ui.b_sliderSelect.clicked.connect(self.__select__)

def main():
    app = QtWidgets.QApplication([])
    window = ui_window()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()