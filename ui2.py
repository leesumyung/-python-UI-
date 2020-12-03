import cv2 as cv
import tkinter
import tkinter.filedialog
from PIL import Image, ImageTk
import numpy as np
import jieba
import data_input_helper as data_helpers

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

#综合图像识别和文字分类
import numpy as np
import os, sys
sys.path.append('textcnn')
from textcnn.predict import RefuseClassification
from classify_image import *
import numpy as np
import os, sys
sys.path.append('textcnn')
from textcnn.predict import RefuseClassification
from classify_image import *
import warnings
warnings.filterwarnings("ignore")

window = tkinter.Tk()
window.title('垃圾分类识别界面')
window.geometry('350x400')

#下面这个可以展开全屏
window.state("zoomed")

#固定窗口，使界面不可放大或缩小
#window.resizable(0, 0)
#var1 = tkinter.StringVar()
#弄个花里花俏一点的界面增加视觉效果而已
#T = tkinter.Label(window, text="25+100=", textvariable=var1, bg="lightGreen", fg="DimGray", anchor="se")
#T.place(x=0, y=0, width=350, height=120)

#显示图片路径以及识别结果的窗口
tkinter.Label(window, text='请输入识别文本: ', font=("微软雅黑", 20)).place(x=40, y=60)
tkinter.Label(window, text='文本识别结果为: ', font=("微软雅黑", 20)).place(x=40, y=120)

var_user_name = tkinter.StringVar()
entry_user_name = tkinter.Entry(window, textvariable=var_user_name, font=("微软雅黑", 15))
entry_user_name.place(x=250, y=65, width=300, height=30)
var_user_pd = tkinter.StringVar()
entry_user_pd = tkinter.Entry(window, textvariable=var_user_pd, font=("微软雅黑", 15))
entry_user_pd.place(x=250, y=125, width=300, height=30)

'''
var_load2 = tkinter.StringVar()
entry_load2 = tkinter.Entry(window, textvariable=var_load2, font=("微软雅黑", 20))
entry_load2.place(x=1060, y=370, width=400, height=50)
var_s2 = tkinter.StringVar()
entry_s2 = tkinter.Entry(window, textvariable=var_s2, font=("微软雅黑", 20))
entry_s2.place(x=1060, y=470, width=400, height=50)
'''

FLAGS = None

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

class RafuseRecognize():

    def __init__(self):
        self.refuse_classification = RefuseClassification()
        self.init_classify_image_model()
        self.node_lookup = NodeLookup(uid_chinese_lookup_path='./data/imagenet_2012_challenge_label_chinese_map.pbtxt',
                                      model_dir='/tmp/imagenet')

    def init_classify_image_model(self):
        create_graph('/tmp/imagenet')

        self.sess = tf.Session()
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('softmax:0')

    def recognize_image(self, image_data):
        predictions = self.sess.run(self.softmax_tensor,
                                    {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]
        result_list = []
        for node_id in top_k:
            human_string = self.node_lookup.id_to_string(node_id)
            # print(human_string)
            human_string = ''.join(list(set(human_string.replace('，', ',').split(','))))
            # print(human_string)
            classification = self.refuse_classification.predict(human_string)
            result_list.append('%s  =>  %s   ' % (human_string, classification))

        return '\n'.join(result_list)

# 打开文件函数
def choose_fiel():
    selectFileName = tkinter.filedialog.askopenfilename(title='选择文件')  # 选择文件
    var_load.set(selectFileName)


# 识别图片数字函数
def main(img):
    test = RafuseRecognize()
    image_data = tf.gfile.FastGFile(img, 'rb').read()
    res = test.recognize_image(image_data)
    var_s.set(res)

def main1():
    test = RefuseClassification()
    res = test.predict(entry_user_pd.get())
    var_user_pd.set(res)

# 清零函数，不过没啥意义，就想把界面弄的对称一点而已
def delete():  # 删除函数
    content = var_user_pd.get()
    var_user_pd.set(content[0:len(content) - 1])


# 显示所要识别的图片函数
def showImg(img1):
    # canvas = tkinter.Canvas(window, height=400, width=1000)
    load = Image.open(img1)
    render = ImageTk.PhotoImage(load)
    img = tkinter.Label(image=render)
    img.image = render
    # canvas.create_image(0, 0, anchor='nw', image=render)
    # canvas.pack(side='bottom')
    img.place(x=160, y=120)


# 按钮
'''
submit_button = tkinter.Button(window, text="选择文件", command=choose_fiel).place(x=50, y=250)
submit_button = tkinter.Button(window, text="文字识别", command=delete()).place(x=50, y=300)
submit_button = tkinter.Button(window, text="显示图片", command=lambda: showImg(entry_user_name.get())).place(x=219, y=250)
submit_button = tkinter.Button(window, text="图片识别", command=lambda: main(entry_user_name.get())).place(x=219, y=300)
'''
submit_button = tkinter.Button(window, text="文本识别", font=("微软雅黑", 20), command=lambda: main1()).place(x=250, y=230)



window.mainloop()