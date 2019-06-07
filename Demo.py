from PIL import Image
from torchvision import transforms
import numpy as np
import torch
from skimage.color import lab2rgb, rgb2gray
import matplotlib.pyplot as plt
from torch.autograd import Variable
from colornet import ColorNet

# 配置cuda
have_cuda = torch.cuda.is_available()
color_model = ColorNet()                                   # 网络架构
color_model.load_state_dict(torch.load('colornet_params.pkl'))    # 参数路径
if have_cuda:
    color_model.cuda()
color_model.eval()


# 处理图像,转成黑白和彩色，供GUI调用
def Picture(name):
    img_name = name  # 输入图片的路径
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

    # 阔展成4维
    original_img = img_original.unsqueeze(0).float()
    original_img = original_img.unsqueeze(1).float()

    print(original_img.size())
    gray_name = 'D:/数据集/生成的图像/' + 'gray' + '.jpg'  # 生成黑白图像的路径
    pic = original_img.squeeze().numpy()
    pic = pic.astype(np.float64)
    plt.imsave(gray_name, pic, cmap='gray')                # 保存生成的黑白图片
    print(original_img.size())
    w = original_img.size()[2]
    h = original_img.size()[3]

    scale_img = img_scale.unsqueeze(0).float()
    scale_img = scale_img.unsqueeze(1).float()

    if have_cuda:                 # 调用设置cuda
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
    color_name = 'D:/数据集/生成的图像/' + 'colored' + '.jpg'  # 生成彩色图像的路径
    plt.imsave(color_name, color_img)

    return gray_name, color_name               # 返回黑白、上色图片的路径


def Picturestyle(name, namestyle):                      # 供风格迁移使用的函数
    img_name = name  # 输入图片的路径
    img = Image.open(img_name)
    scale_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),  # 剪切图片大小为224*224
    ])
    img1 = scale_transform(img)
    img_scale = np.asarray(img1)
    img_original = np.asarray(img)

    img_style = namestyle                    # 风格迁移中另一张图片的路径
    style_img = Image.open(img_style)
    scale_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),  # 剪切图片大小为224*224
    ])
    img2 = scale_transform(style_img)

    style = np.asarray(img2)           # 按同样方法处理风格迁移的图片
    style = rgb2gray(style)
    style = torch.from_numpy(style)
    style = style.unsqueeze(0).float()
    style = style.unsqueeze(1).float()

    img_scale = rgb2gray(img_scale)
    img_scale = torch.from_numpy(img_scale)
    img_original = rgb2gray(img_original)
    img_original = torch.from_numpy(img_original)
    print(img_original.size())

    # 阔展成4维
    original_img = img_original.unsqueeze(0).float()
    original_img = original_img.unsqueeze(1).float()

    print(original_img.size())
    gray_name = 'D:/数据集/生成的图像/' + 'gray1' + '.jpg'  # 生成黑白图像的路径
    pic = original_img.squeeze().numpy()
    pic = pic.astype(np.float64)
    plt.imsave(gray_name, pic, cmap='gray')
    print(original_img.size())
    w = original_img.size()[2]
    h = original_img.size()[3]

    scale_img = img_scale.unsqueeze(0).float()
    scale_img = scale_img.unsqueeze(1).float()

    if have_cuda:
        original_img, scale_img, style = original_img.cuda(), scale_img.cuda(), style.cuda()
    with torch.no_grad():
        original_img, scale_img, style = Variable(original_img), Variable(scale_img),Variable(style)
    # 输入网络，scale_image进入全局特征提取的网络
    _, output = color_model(original_img, style)

    color_img = torch.cat((original_img, output[:, :, 0:w, 0:h]), 1)  # L与ab融合
    print(color_img.size())
    color_img = color_img.data.cpu().numpy().transpose((0, 2, 3, 1))  # 转置

    print(type(color_img))

    color_img = color_img[0]
    color_img[:, :, 0:1] = color_img[:, :, 0:1] * 100
    color_img[:, :, 1:3] = color_img[:, :, 1:3] * 255 - 128
    color_img = color_img.astype(np.float64)
    color_img = lab2rgb(color_img)
    color_name = 'D:/数据集/生成的图像/' + 'colored' + '.jpg'  # 生成彩色图像的路径
    plt.imsave(color_name, color_img)

    return color_name             # 返回生成图片的路径



