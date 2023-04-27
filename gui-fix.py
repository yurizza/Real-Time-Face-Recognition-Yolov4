import sys

import cv2
import numpy as np
import time

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.uic import loadUi
import datetime

with open('obj.names') as f:
    labels = [line.strip() for line in f]
with open('nama.txt') as f:
    text = [line.strip() for line in f]
with open('nim.txt') as f:
    textNim = [line.strip() for line in f]

network = cv2.dnn.readNetFromDarknet('yolov4-obj.cfg',
                                     'yolov4-obj_last.weights')
network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)

network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layers_names_all = network.getLayerNames()

layers_names_output = \
    [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]

probability_minimum = 0.96

threshold = 0.3

class presensi(QMainWindow):
    def __init__(self):
        super(presensi,self).__init__()

        loadUi('coba2.ui', self)
        self.logic=0
        self.open.clicked.connect(self.onClicked)
        self.Confirm.clicked.connect(self.ConfirmClicked)

    @pyqtSlot()
    def onClicked(self):
        if self.logic==0:
            cap = cv2.VideoCapture(0)

            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    self.displayImage(frame, 1)
                    cv2.waitKey(10)
                    if(self.logic==2):
                        cap.release()
                else:
                    print('return not found')
            cap.release()

            cv2.destroyAllWindows()

    def ConfirmClicked(self):
        self.close()


    def displayImage(self, img, window=1):
        qformat = QImage.Format.Format_Indexed8

        if len(img.shape)==3:
            if(img.shape[2])==4:
                qformat=QImage.Format_RGBA888
            else:
                qformat=QImage.Format_RGB888
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
        network.setInput(blob)  
        start = time.time()
        output_from_network = network.forward(layers_names_output)
        end = time.time()
        print('Current frame took {:.5f} seconds'.format(end - start))
        
        bounding_boxes = []
        confidences = []
        class_numbers = []

        for result in output_from_network:

            for detected_objects in result:
                
                scores = detected_objects[5:]
                
                class_current = np.argmax(scores)
                
                confidence_current = scores[class_current]

                if confidence_current > probability_minimum:
                   
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])

                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    bounding_boxes.append([x_min, y_min,
                                        int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)

        if len(results) > 0:
            for i in results.flatten():
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                cv2.rectangle(img, (x_min, y_min),
                            (x_min + box_width, y_min + box_height), (40, 198, 31), 3)

                text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                    confidences[i])
                                                    
                self.name.setText(text[int(class_numbers[i])])
                self.nim.setText(textNim[int(class_numbers[i])])
                
                cv2.putText(img, text_box_current, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        img = QImage(img, img.shape[1], img.shape[0], qformat)
        img=img.rgbSwapped()
        self.imgLabel.setPixmap(QPixmap.fromImage(img))
        self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

app=QApplication(sys.argv)
window=presensi()
window.show()

try:
    sys.exit(app.exec_())
except:
    print('exiting')
