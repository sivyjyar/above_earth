from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import math
import time
from scipy import interpolate


def chunks(lst, count):
    start = 0
    for i in range(count):
          stop = start + len(lst[i::count])
          yield lst[start:stop]
          start = stop     

#width_window = 100
#height_window = 50

width_window = 200.0
height_window = 500.0


def crop_rect(img, rect, ix):
    print("rect!")
    center, size, angle = rect[0], rect[1], rect[2]

    center, size = tuple(map(int, center)), tuple(map(int, size))
    height, width = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D(center, angle, 1)

    img_rot = cv2.warpAffine(img[:,:,:3], M, (width, height))
    # cv2.imshow('da', img_rot)
    # time.sleep(2)
    cut_img = cv2.getRectSubPix(img_rot, size, center)

    
    # сохранить часть изображения
    ix1 = ix + 780
    cv2.imwrite(f'test/shit{ix}.jpg', cut_img)
    
    cv2.imshow('cut_img', cut_img)
    # time.sleep(0.08)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()

#    for h in range(img_rot.shape[0]):
#        for w in range(img_rot.shape[1]):
#            if img_rot[h,w] == 0:
#                img[h,w,:] = 255
#     print ("CROP--->", img_rot.shape, cut_img.shape)
    return img


class CutTool():
    def __init__(self, master):
        # найстройки основного окна
        self.parent = master
        self.parent.title("CutTool")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width = True, height = True)

        # инициализировать состояние мыши
        self.STATE = {}
        self.STATE['click'] = 0
        self.STATE['x'], self.STATE['y'] = 0, 0

        # ссылка на координаты 
        self.coordList = []
        self.running = False
        
        self.image_container = False
        
        self.step_size = 20
        # ----------------- GUI stuff ---------------------

        # главная панель для маркировки
        self.mainPanel = Canvas(self.frame, cursor='tcross')
#        self.mainPanel.bind("<Button-1>", self.mouseClick)
#        self.mainPanel.bind("<ButtonPress-1>", self.mouseClick)
        self.mainPanel.bind('<ButtonPress-1>',self.start_motor)
        self.mainPanel.bind('<ButtonRelease-1>',self.stop_motor)
        
        
        self.mainPanel.bind("<Motion>", self.mouseMove)
        self.mainPanel.grid(row = 2, column = 1, rowspan = 4, sticky = W+N)

        # панель управления для навигации по изображениям
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row = 6, column = 1, columnspan = 2, sticky = W+E)

        # отображение положения мыш
        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side = RIGHT)
        
        self.loadImage()
        
    def mouseClick(self, event):
        if self.STATE['click'] == 0:
            self.STATE['x'], self.STATE['y'] = event.x, event.y
        print ("---------", self.STATE)

    def mouseMove(self, event):
        if self.running:
            self.draw(event)
        self.disp.config(text = 'x: %d, y: %d' %(event.x, event.y))

    def start_motor(self, event):
        self.running = True
        print("starting motor...")
    
    def stop_motor(self, event):
        print("stopping motor...")
        self.running = False
        self.curve_2d(self.coordList)
        self.coordList = []

    def curve_2d(self, x):
        R = len(x)
        pts = x
        x = np.array(x)
        ptdiff = lambda p1, p2: (p1[0]-p2[0], p1[1]-p2[1])

        diffs = (ptdiff(p1, p2) for p1, p2 in zip (pts, pts[1:]))
        path = sum(math.hypot(*d) for d in  diffs)
    #    test = chunks(pts, 4)
    #    print(path, R//4)

        img_array = np.array(self.img)
        img_array2 = np.array(self.img)
        # print (img_array.shape)
        global img_

        a0 = [self.step_size * x for x in range(4, int((self.img.size[1]-100) // 20 ))]
        a1 = np.array(a0)
        # print(self.img.size[1])
        # print("a1=", a1)
        # print(a1[6:-6])
        print(a1)
        for i, ix in enumerate(a1):
            print(i, ix)


        for ix, i in enumerate(a1):
            if ix == 0: continue
            if i!=a1[-1]:
                """
                1 узнать угол между настоящей точкой и прошлой по отношению к оси 'x'
                2 провернуть на этот угол точки для получения прямоугольника 
                """
                # a = np.array(pts[ix])
                # b = np.array(pts[ix-self.step_size])

                # print(a1)
                # print(range(1, int(self.img.size[1]/2//20)), self.img.size[1])
                
                # ab = a - b
                #
                # hypotenuse = np.hypot(ab[0], ab[1])
                #
                # cos_A = ab[0]/hypotenuse
                # angle = int(math.degrees(math.acos(cos_A)))
                #
#                theta = np.radians(int(angle))
# #                r = np.array(( (np.cos(theta), -np.sin(theta)),
# #                               (np.sin(theta),  np.cos(theta)) ))
#
#                 if ab[1] < 0:
#                     angle = -float(angle)
#                 else:
#                     angle = float(angle)
#                rect_ = ((float(a[0]), float(a[1])), (100.0, 200.0), angle)
#                 rect_ = ((float(a[0]), float(a[1])), (height_window, width_window), angle)

                rect_ = ((float(self.img.size[0]/2), float(a1[ix])), (height_window, width_window), 00)

                print ("----->", rect_)
#                rect = ((a[0], a[1]), (100, 200), angle_)

                # вырезать кусок картинки из общего изображения
    
                img_array2_ = crop_rect(img_array, rect_, ix)


#                b_rect = [(a[0], a[1]-100),
#                          (a[0]+100, a[1]-100),
#                          (a[0], a[1]+100),
#                          (a[0]+100, a[1]+100)]
#                rect = cv2.minAreaRect(np.array(b_rect))
#
#                 box = cv2.boxPoints(rect_)
#                 box = np.int0(box)
#                 cv2.drawContours(img_array2, [box], 0, (0,0,255), 2)
#                 cv2.circle(img_array2, (a[0], a[1]), radius=10, color=(0, 0, 255), thickness=-1)
                
#                cv2.circle(img_array2, (a[0], a[1]+100), radius=8, color=(0, 0, 255), thickness=-1) # 3
#                cv2.circle(img_array2, (a[0], a[1]-100), radius=8, color=(0, 0, 255), thickness=-1) # 1
#                cv2.circle(img_array2, (a[0]+100, a[1]+100), radius=8, color=(0, 0, 255), thickness=-1) #4
#                cv2.circle(img_array2, (a[0]+100, a[1]-100), radius=8, color=(0, 0, 255), thickness=-1) # 2
                
#                print (ab, hypotenuse, angle, rect, box)
                #------------------------->
                # по точке на изображения вырезаю картинку
#                cut_img = img_array[pts[ix][1]-(width_window//2):pts[ix][1]+(width_window//2), 
#                                   pts[ix][0]-(height_window//2):pts[ix][0]+(height_window//2), :]                
#                img_array2[pts[ix][1]-(width_window//2):pts[ix][1]+(width_window//2), 
#                          pts[ix][0]-(height_window//2):pts[ix][0]+(height_window//2), :] = 255    

                # img_ = ImageTk.PhotoImage(image=Image.fromarray(img_array2))
                #
                # self.mainPanel.itemconfig(self.image_container, image=img_)
                # # обновить канвас
                # self.mainPanel.update()

            else:
                print(ix % self.step_size,ix ,self.step_size, 'lox')
                break
        
                            

    def loadImage(self):
        # загрузить изображение

        self.img = Image.open("map_1.png")
        size = self.img.size
        self.factor = max(size[0]/10000., size[1]/10000., 1.)
        self.img = self.img.resize((int(size[0]/self.factor), int(size[1]/self.factor)))
        print (int(size[0]/self.factor), int(size[1]/self.factor))
        self.tkimg = ImageTk.PhotoImage(self.img)
        self.mainPanel.config(width = max(self.tkimg.width(), 400), height = max(self.tkimg.height(), 400))
        self.image_container = self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)
        
    def draw(self, event):
        self.coordList.append((event.x, event.y))
        self.mainPanel.create_oval(event.x - 3,
                                  event.y - 3,
                                  event.x + 3,
                                  event.y + 3,
                                  fill="red", outline="red")

if __name__ == '__main__':
    root = Tk()
    tool = CutTool(root)
    root.resizable(width =  True, height = True)
    root.mainloop()
