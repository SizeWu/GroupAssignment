import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Demo import Picture
from PIL import Image
from torchvision import transforms
from skimage.color import lab2rgb, rgb2gray
from torch.autograd import Variable
from colornet import ColorNet
#存储路径
PATH="C:/Users/dell/Desktop/python_hw3/"
#配置cuda
have_cuda = torch.cuda.is_available()
color_model = ColorNet()
# 参数路径
color_model.load_state_dict(torch.load('colornet_params2.pkl'))
#有cuda，则使用cuda
if have_cuda:
    color_model.cuda()
color_model.eval()
# 处理图像,转成黑白和彩色
def Picture(name):

    img_name = PATH+'second_image/'+name  # 输入图片的路径
    img = Image.open(img_name)
    scale_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),  # 剪切图片大小为224*224
    ])
    img1 = scale_transform(img)
    img_scale = np.asarray(img1)
    img_original = np.asarray(img)
    img_scale = rgb2gray(img_scale)
    img_scale = torch.from_numpy(img_scale)
    img_original = rgb2gray(img_original)
    img_original = torch.from_numpy(img_original)
    print(img_original.size())
    # 扩展成4维
    original_img = img_original.unsqueeze(0).float()
    original_img = original_img.unsqueeze(1).float()
    print(original_img.size())
    gray_name = PATH+'gray_image/'+name  # 生成黑白图像的路径
    pic = original_img.squeeze().numpy()
    pic = pic.astype(np.float64)
    plt.imsave(gray_name, pic, cmap='gray')
    print(original_img.size())
    w = original_img.size()[2]
    h = original_img.size()[3]

    scale_img = img_scale.unsqueeze(0).float()
    scale_img = scale_img.unsqueeze(1).float()

    if have_cuda:
        original_img, scale_img = original_img.cuda(), scale_img.cuda()
    with torch.no_grad():
        original_img, scale_img = Variable(original_img), Variable(scale_img)
    # 输入网络，scale_image进入全局特征提取的网络，若要做风格迁移，则scale_image改成另一个图片
    _, output = color_model(original_img, scale_img)

    color_img = torch.cat((original_img, output[:, :, 0:w, 0:h]), 1)  # L与ab融合
    print(color_img.size())
    color_img = color_img.data.cpu().numpy().transpose((0, 2, 3, 1))  # 转置

    print(type(color_img))

    color_img = color_img[0]
    color_img[:, :, 0:1] = color_img[:, :, 0:1] * 100
    color_img[:, :, 1:3] = color_img[:, :, 1:3] * 255 - 128
    color_img = color_img.astype(np.float64)
    color_img = lab2rgb(color_img)
    color_name = PATH+'pic_image/'+name   # 生成彩色图像的路径
    plt.imsave(color_name, color_img)

    return gray_name, color_name

def get_size(file):
    # 获取文件大小:KB
    size = os.path.getsize(file)
    return size / 1024
#压缩图片至指定大小
def compress_image( infile , outfile , mb=40, step=10, quality=80):
    """不改变图片尺寸压缩到指定大小
    :param infile: 压缩源文件
    :param outfile: 压缩文件保存地址
    :param mb: 压缩目标，KB
    :param step: 每次调整的压缩比率
    :param quality: 初始压缩比率
    :return: 压缩文件地址，压缩文件大小
    """
    o_size = get_size(infile)
    #判断是否需要压缩文件
    if o_size <= mb:
        im = Image.open(infile)
        im.save(outfile)
        return outfile
    while o_size > mb:
        im = Image.open(infile)
        im.save(outfile, quality=quality)
        #取像素点
        if quality - step < 0:
            break
        quality -= step
        o_size = get_size(outfile)
    return outfile
#将视频每一帧保存为图片并压缩
def Capture(path1, path2='first_image/', path3='second_image/'):

    videoCapture = cv2.VideoCapture()
    #打开视频
    videoCapture.open(PATH+'pre_video/'+path1)
    #获取帧率
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    # fps是帧率，意思是每一秒刷新图片的数量，frames是一整段视频中总的图片数量。
    print("fps=", fps, "frames=", frames)
    #将每一张图片压缩并保存
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        cv2.imwrite(PATH+path2 + '/%d.jpg' % i, frame)
        compress_image(PATH+path2 + '/%d.jpg' % i,PATH+path3 + '/%d.jpg' % i)
        Picture('/%d.jpg' % i)
    else:
        pass
#将黑白图像与上色图像拼接
def conj_pic(i):
    im_list = []
    # 获取相同序数的图片
    im_list.append(Image.open(PATH+'gray_image/'+'%d.jpg'%i))
    im_list.append(Image.open(PATH+'pic_image/'+'%d.jpg'%i))
    ims = []
    #将每张图片处理为相同尺寸
    for x in im_list:
        new_img = x.resize((1280,720), Image.BILINEAR)
        ims.append(new_img)
    # 单幅图像尺寸
    width, height = ims[0].size
    # 创建空白长图
    result = Image.new(ims[0].mode, (width * len(ims), height) )
    # 拼接图片
    for j, im in enumerate(ims):
        result.paste(im, box=(j*width ,0))
    # 保存图片
    result.save(PATH+'conj_image/'+'%d.jpg'%i)
#品阶所有图片
def conj_all():
    filelist = os.listdir(PATH + 'gray_image/')
    num = len(filelist)
    for i in range(num):
        conj_pic(i)
#将图片连接为视频
def Fusion(path):
    #获取文件夹中所有图片
    filelist = os.listdir(PATH+path)
    num = len(filelist)
    #设置帧率
    fps = 30
    pic = PATH+path+'/'+filelist[0]
    img = cv2.imread(pic)
    imgInfo = img.shape
    size = (imgInfo[1], imgInfo[0])  # 宽度和高度信息
    print(size)
    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
    #设置视频格式
    video = cv2.VideoWriter(filename=PATH+'result.avi', fourcc=fourcc, fps=fps, frameSize=size)
    #将照片逐帧写入视频
    for i in range(num):
            item = PATH+path +'/'+ '%d.jpg' % i
            img = cv2.imread(item)
            video.write(img)
    video.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    path1 = 'cideo_Trim.mp4'
#    path2 = 'E:/video'
    path3 = "conj_image"
#    Capture(path1)
#    conj_all()
    Fusion(path3)

