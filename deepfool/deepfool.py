import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")

    # 这里做一个前向传播，然后计算出原始图片的predict
    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]    # label 是正确的预测分类标签， 神经网络对未处理的原始图片的分类

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    # 初始时候 pert 是 image 的深copy
    x = Variable(pert_image[None, :], requires_grad=True)
    # print("x shape:{0:}".format(x.shape))
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    print(fs[0,I[k]])
    k_i = label

    # 小于迭代次数 或者类别不同，就停止迭代，最大的迭代次数为50次
    while k_i == label and loop_i < max_iter:   

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        # 选择一个最近的超平面
        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)  # r_i 是到超平面的距离
        r_tot = np.float32(r_tot + r_i)  # 加上上一次的距离

        # print("rot shape: {0:}".format(r_tot.shape))  # r_tot 是扰动， 所有维度距离向量的距离
        # 产生攻击图片， 用原来的image + 一些扰动
        # rot 是指导 分界面的距离， 稍微加上一点，就可以产生误分类
        # overshoot 就是用于越过分界面， overshoot 一直没有变，为 0.02
        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())   # 这里是产生最后的预测,相当于 label  

        loop_i += 1  # 循环次数

    r_tot = (1 + overshoot) * r_tot   # overshoot = 0.02

    # 返回的四个值分别为 r_tot 到分类面的距离， loop_i 迭代循环的次数  label 原图预测  k_i 攻击图片预测
    # pert_iamge  攻击图片
    return r_tot, loop_i, label, k_i, pert_image

