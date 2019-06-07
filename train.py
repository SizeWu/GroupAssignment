import os
import traceback
import torch
import torch.nn.functional as F         # 导入必须的模型，这里比较重要的是from . import functional as F，也就是导入了
                                        # functional.py脚本中具体的data augmentation函数
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

from myimgfolder import TrainImageFolder
from colornet import ColorNet


original_transform = transforms.Compose([
    transforms.Resize(256),                              # 将输入的`PIL.Image`重新改变大小size，size是最小边的边长
                                                        # 目前已经被transforms.Resize类取代了

    transforms.RandomCrop(224),                         # 依据给定的size随机裁剪,在这种情况下，切出来的图片的形状是正方形
    transforms.RandomHorizontalFlip(),                  # 随机水平翻转给定的PIL.Image,翻转的概率为0.5。

    #transforms.ToTensor()                              # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
])

have_cuda = torch.cuda.is_available()
epochs = 10
data_dir = "/input_dir/datasets/Caltech256/256_ObjectCategories"
# data_dir = "../images256/"
train_set = TrainImageFolder(data_dir, original_transform)    # 建训练集
train_set_size = len(train_set)
train_set_classes = train_set.classes                         # classes (list): List of the class names.
print('train_set_classes', train_set_classes)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4)
color_model = ColorNet()
if os.path.exists('/output_dir/colornet_paramsCaltech257.pkl'):
    color_model.load_state_dict(torch.load('/output_dir/colornet_paramsCaltech257.pkl'))
if have_cuda:
    color_model.cuda()
optimizer = optim.Adadelta(color_model.parameters())            # 优化方案：Adadelta


def train(epoch):
    color_model.train()

    try:
        for batch_idx, (data, classes) in enumerate(train_loader):
            messagefile = open('/output_dir/message.txt', 'a')
            original_img = data[0].unsqueeze(1).float()  # 在第一维增加一个维度
            img_ab = data[1].float()
            if have_cuda:
                original_img = original_img.cuda()
                img_ab = img_ab.cuda()
                classes = classes.cuda()
            original_img = Variable(original_img)
            img_ab = Variable(img_ab)
            classes = Variable(classes)
            optimizer.zero_grad()                       # 将梯度初始化为零（因为一个batch的loss关于weight的导数是所有
                                                        # sample的loss关于weight的导数的累加和）
            # with torch.no_grad():                       # GPU分配有问题，加上这行代码
            class_output, output = color_model(original_img, original_img)  # 前向传播求出预测的值
            ems_loss = torch.pow((img_ab - output), 2).sum() / torch.from_numpy(np.array(list(output.size()))).prod()
            cross_entropy_loss = 1/300 * F.cross_entropy(class_output, classes)  # 交叉熵误差函数，考虑了全局信息
            loss = ems_loss + cross_entropy_loss    # 全局信息的比重是不是可以考虑调一下？
            lossmsg = 'loss: %.9f\n' % (loss.item())
            messagefile.write(lossmsg)
            ems_loss.backward(retain_graph=True)   # 保留计算图
            cross_entropy_loss.backward()
            optimizer.step()    # 更新所有参数
            if batch_idx % 500 == 0:
                message = 'Train Epoch:%d\tPercent:[%d/%d (%.0f%%)]\tLoss:%.9f\n' % (
                    epoch, batch_idx * len(data), 2*len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item())
                messagefile.write(message)
                torch.save(color_model.state_dict(), '/output_dir/colornet_paramsCaltech257.pkl')
                print(message)
            messagefile.close()

    except Exception:
        logfile = open('/output_dir/log.txt', 'w')
        logfile.write(traceback.format_exc())
        logfile.close()
    finally:
        torch.save(color_model.state_dict(), '/output_dir/colornet_paramsCaltech257.pkl')


for epoch in range(1, epochs + 1):
    train(epoch)
