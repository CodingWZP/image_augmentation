# image_augmentation
Image augmentation and change bounding box at the same time.

# 安装依赖库和imgaug库
在训练yolo的时候，对已有数据集已经打好了标签，想要进行数据的增强（数据扩增），可以通过imgaug实现对图片和标签中的boundingbox同时变换。
[imgaug使用文档](https://imgaug.readthedocs.io/en/latest/index.html)
首先，安装依赖库。
```python
pip install six numpy scipy matplotlib scikit-image opencv-python imageio`
```
安装imgaug

```python
pip install imgaug
```

# Bounding Boxes实现
## 读取原影像bounding boxes坐标
读取xml文件并使用ElementTree对xml文件进行解析，找到每个object的坐标值。

```python
def change_xml_annotation(root, image_id, new_target):
    new_xmin = new_target[0]
    new_ymin = new_target[1]
    new_xmax = new_target[2]
    new_ymax = new_target[3]

    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    object = xmlroot.find('object')
    bndbox = object.find('bndbox')
    xmin = bndbox.find('xmin')
    xmin.text = str(new_xmin)
    ymin = bndbox.find('ymin')
    ymin.text = str(new_ymin)
    xmax = bndbox.find('xmax')
    xmax.text = str(new_xmax)
    ymax = bndbox.find('ymax')
    ymax.text = str(new_ymax)
    tree.write(os.path.join(root, str("%06d" % (str(id) + '.xml'))))
```

## 生成变换序列
产生一个处理图片的Sequential。

```python
# 影像增强
    seq = iaa.Sequential([
        iaa.Invert(0.5),
        iaa.Fliplr(0.5),  # 镜像
        iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
        iaa.GaussianBlur(sigma=(0, 3.0)),  # iaa.GaussianBlur(0.5),
        iaa.Affine(
            translate_px={"x": 15, "y": 15},
            scale=(0.8, 0.95),
        )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])
```

## bounding box 变化后坐标计算
先读取该影像对应xml文件，获取所有目标的bounding boxes，然后依次计算每个box变化后的坐标。

```python
seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变，而不是随机
# 读取图片
img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.jpg'))
# sp = img.size
img = np.asarray(img)
# bndbox 坐标增强
for i in range(len(bndbox)):
    bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3]),
    ], shape=img.shape)

    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    boxes_img_aug_list.append(bbs_aug)

    # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
    n_x1 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x1)))
    n_y1 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y1)))
    n_x2 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x2)))
    n_y2 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y2)))
    if n_x1 == 1 and n_x1 == n_x2:
        n_x2 += 1
    if n_y1 == 1 and n_y2 == n_y1:
        n_y2 += 1
    if n_x1 >= n_x2 or n_y1 >= n_y2:
        print('error', name)
    new_bndbox_list.append([n_x1, n_y1, n_x2, n_y2])
# 存储变化后的图片
image_aug = seq_det.augment_images([img])[0]
path = os.path.join(AUG_IMG_DIR,
                    str(str(name[:-4]) + '_' + str(epoch)) + '.jpg')
image_auged = bbs.draw_on_image(image_aug, thickness=0)
Image.fromarray(image_auged).save(path)

# 存储变化后的XML
change_xml_list_annotation(XML_DIR, name[:-4], new_bndbox_list, AUG_XML_DIR,
                           str(name[:-4]) + '_' + str(epoch))
# print(str(str(name[:-4]) + '_' + str(epoch)) + '.jpg')
new_bndbox_list = []
```

# 使用示例
## 数据准备
输入数据为两个文件夹一个是需要增强的影像数据（JPEGImages），一个是对应的xml文件（Annotations）。==注意：影像文件名需和xml文件名相对应！==
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210325164244643.png)![在这里插入图片描述](https://img-blog.csdnimg.cn/20210325164202912.png)
## 设置文件路径

```python
IMG_DIR = "./VOCdevkit/VOC2007/JPEGImages"
XML_DIR = "./VOCdevkit/VOC2007/Annotations"

AUG_XML_DIR = "./AUG/Annotations"  # 存储增强后的XML文件夹路径
try:
    shutil.rmtree(AUG_XML_DIR)
except FileNotFoundError as e:
    a = 1
mkdir(AUG_XML_DIR)

AUG_IMG_DIR = "./AUG/JPEGImages"  # 存储增强后的影像文件夹路径
try:
    shutil.rmtree(AUG_IMG_DIR)
except FileNotFoundError as e:
    a = 1
mkdir(AUG_IMG_DIR)
```
## 设置增强次数

```python
    AUGLOOP = 10 # 每张影像增强的数量
```
## 设置增强参数
通过修改Sequential函数参数进行设置，具体设置参考[imgaug使用文档](https://imgaug.readthedocs.io/en/latest/index.html)

```python
	seq = iaa.Sequential([
        iaa.Invert(0.5),
        iaa.Fliplr(0.5),  # 镜像
        iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
        iaa.GaussianBlur(sigma=(0, 3.0)),  # iaa.GaussianBlur(0.5),
        iaa.Affine(
            translate_px={"x": 15, "y": 15},
            scale=(0.8, 0.95),
        )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])

```
## 修改xml文件中filename和path

```python
tree = ET.parse(in_file)
#修改xml中的filename
elem = tree.find('filename')
elem.text = (str(id) + '.jpg')
xmlroot = tree.getroot()
#修改xml中的path
elem = tree.find('path')
elem.text = ('C:\working\yolo3\VOCdevkit\VOC2007\JPEGImages\\' + str(id) + '.jpg')
xmlroot = tree.getroot()
```


## 输出
运行augmentation.py ，运行结束后即可得到增强的影像和对应的xml文件夹。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210325164837254.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210325164900829.png)


# 完整代码

```python
import xml.etree.ElementTree as ET
import pickle
import os
from os import getcwd
import numpy as np
from PIL import Image
import shutil
import matplotlib.pyplot as plt

import imgaug as ia
from imgaug import augmenters as iaa


ia.seed(1)


def read_xml_annotation(root, image_id):
    in_file = open(os.path.join(root, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    bndboxlist = []

    for object in root.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([xmin, ymin, xmax, ymax])
        # print(bndboxlist)

    bndbox = root.find('object').find('bndbox')
    return bndboxlist


# (506.0000, 330.0000, 528.0000, 348.0000) -> (520.4747, 381.5080, 540.5596, 398.6603)
def change_xml_annotation(root, image_id, new_target):
    new_xmin = new_target[0]
    new_ymin = new_target[1]
    new_xmax = new_target[2]
    new_ymax = new_target[3]

    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    object = xmlroot.find('object')
    bndbox = object.find('bndbox')
    xmin = bndbox.find('xmin')
    xmin.text = str(new_xmin)
    ymin = bndbox.find('ymin')
    ymin.text = str(new_ymin)
    xmax = bndbox.find('xmax')
    xmax.text = str(new_xmax)
    ymax = bndbox.find('ymax')
    ymax.text = str(new_ymax)
    tree.write(os.path.join(root, str("%06d" % (str(id) + '.xml'))))


def change_xml_list_annotation(root, image_id, new_target, saveroot, id):
    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    #修改xml中的filename
    elem = tree.find('filename')
    elem.text = (str(id) + '.jpg')
    xmlroot = tree.getroot()
    #修改xml中的path
    elem = tree.find('path')
    elem.text = ('C:\working\yolo3\VOCdevkit\VOC2007\JPEGImages\\' + str(id) + '.jpg')
    xmlroot = tree.getroot()

    index = 0
    for object in xmlroot.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        # xmin = int(bndbox.find('xmin').text)
        # xmax = int(bndbox.find('xmax').text)
        # ymin = int(bndbox.find('ymin').text)
        # ymax = int(bndbox.find('ymax').text)

        new_xmin = new_target[index][0]
        new_ymin = new_target[index][1]
        new_xmax = new_target[index][2]
        new_ymax = new_target[index][3]

        xmin = bndbox.find('xmin')
        xmin.text = str(new_xmin)
        ymin = bndbox.find('ymin')
        ymin.text = str(new_ymin)
        xmax = bndbox.find('xmax')
        xmax.text = str(new_xmax)
        ymax = bndbox.find('ymax')
        ymax.text = str(new_ymax)

        index = index + 1

    tree.write(os.path.join(saveroot, str(id + '.xml')))


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


if __name__ == "__main__":

    IMG_DIR = "./VOCdevkit/VOC2007/JPEGImages"
    XML_DIR = "./VOCdevkit/VOC2007/Annotations"

    AUG_XML_DIR = "./AUG/Annotations"  # 存储增强后的XML文件夹路径
    try:
        shutil.rmtree(AUG_XML_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_XML_DIR)

    AUG_IMG_DIR = "./AUG/JPEGImages"  # 存储增强后的影像文件夹路径
    try:
        shutil.rmtree(AUG_IMG_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_IMG_DIR)

    AUGLOOP = 10  # 每张影像增强的数量

    boxes_img_aug_list = []
    new_bndbox = []
    new_bndbox_list = []

    # 影像增强
    seq = iaa.Sequential([
        iaa.Invert(0.5),
        iaa.Fliplr(0.5),  # 镜像
        iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
        iaa.GaussianBlur(sigma=(0, 3.0)),  # iaa.GaussianBlur(0.5),
        iaa.Affine(
            translate_px={"x": 15, "y": 15},
            scale=(0.8, 0.95),
        )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])

    for root, sub_folders, files in os.walk(XML_DIR):

        for name in files:

            bndbox = read_xml_annotation(XML_DIR, name)
            shutil.copy(os.path.join(XML_DIR, name), AUG_XML_DIR)
            shutil.copy(os.path.join(IMG_DIR, name[:-4] + '.jpg'), AUG_IMG_DIR)

            for epoch in range(AUGLOOP):
                seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变，而不是随机
                # 读取图片
                img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.jpg'))
                # sp = img.size
                img = np.asarray(img)
                # bndbox 坐标增强
                for i in range(len(bndbox)):
                    bbs = ia.BoundingBoxesOnImage([
                        ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3]),
                    ], shape=img.shape)

                    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
                    boxes_img_aug_list.append(bbs_aug)

                    # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
                    n_x1 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x1)))
                    n_y1 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y1)))
                    n_x2 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x2)))
                    n_y2 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y2)))
                    if n_x1 == 1 and n_x1 == n_x2:
                        n_x2 += 1
                    if n_y1 == 1 and n_y2 == n_y1:
                        n_y2 += 1
                    if n_x1 >= n_x2 or n_y1 >= n_y2:
                        print('error', name)
                    new_bndbox_list.append([n_x1, n_y1, n_x2, n_y2])
                # 存储变化后的图片
                image_aug = seq_det.augment_images([img])[0]
                path = os.path.join(AUG_IMG_DIR,
                                    str(str(name[:-4]) + '_' + str(epoch)) + '.jpg')
                image_auged = bbs.draw_on_image(image_aug, thickness=0)
                Image.fromarray(image_auged).save(path)

                # 存储变化后的XML
                change_xml_list_annotation(XML_DIR, name[:-4], new_bndbox_list, AUG_XML_DIR,
                                           str(name[:-4]) + '_' + str(epoch))
                # print(str(str(name[:-4]) + '_' + str(epoch)) + '.jpg')
                new_bndbox_list = []
```
