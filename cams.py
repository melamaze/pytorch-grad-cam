import argparse
import cv2
import numpy as np
import torch
import glob as glob
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise
    

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.nn import functional as F
from torch import topk
from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive

# from models import CNN_Model
# from models.DLA import SimpleDLA
from models.resnet import ResNet18
# from models.VGG16 import VGG
from models.macnn import MACNN


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    # parser.add_argument(
    #     '--image-path',
    #     type=str,
    #     default='./examples/both.png',
    #     help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'hirescam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')
    parser.add_argument('--folder', type=str, help='It will process all the pictures in this folder')
    # 攻擊者模式: --attacker=1
    parser.add_argument('--attacker',default=0, type=int, help='determine trigger or not. 0 is not.')


    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "hirescam":HiResCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad,
         "gradcamelementwise": GradCAMElementWise}

    # model = models.resnet50(pretrained=True)
    
    # train好的model
    PATH = 'models/vgg19.pth'

    # 這個好像沒差，應該是不用先save = =
    # torch.save(CNN_Model().state_dict(), PATH)

    model = MACNN()
    # model = ResNet18()
    model.eval()
    model.load_state_dict(torch.load(PATH, map_location='cpu'))
    # model = model.vgg
    print([model])

    

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    
    # print(find_layer_types_recursive(model, [torch.nn.Conv2d]))
    # target_layers = model.features[-1]

    # 可以任意更換要找的layer: AdaptiveAvgPool2d, Conv2d, ReLU
    target_layers = find_layer_types_recursive(model, [torch.nn.Conv2d])
    
    print(target_layers)
    # print('in', args.folder)
    # folder = args.folder + '/*'

    poisonCNT = 0
    failCNT = 0
    otherCNT = 0
    totalCNT = 0

    # for image_path in glob.glob(folder):
        
        # rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        # rgb_img = cv2.resize(rgb_img, (32, 32), interpolation=cv2.INTER_AREA)
        # rgb_img = np.float32(rgb_img) / 255
        # input_tensor = preprocess_image(rgb_img,
        #                                 mean=[0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225])
        # print(rgb_img.shape)

    trans_setting = transforms.Compose([
        transforms.ToTensor(), # 轉為 Tensor
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # 灰階轉為 RGB
    ])
    # 正常拿 batch 資料的方式
    # dataset_train = torch.utils.data.DataLoader (datasets.MNIST('../data/mnist/', train=True, download=True,transform=trans_setting),batch_size=10)
    
    # 現在的方式: 一張一張拿，複製 10 張包成一個 batch
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,transform=trans_setting)
        # dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,  transform=transforms.ToTensor())
       
    
    # for batch_idx, (images, labels) in enumerate(dataset_train):
    for i  in range(len(dataset_train)):
        print(i)
        image, label = dataset_train[i]

        # 攻擊者在右下角加上 trigger
        if args.attacker==1:
            image[0][27][26] = 1.0
            image[0][27][27] = 1.0
            image[0][26][26] = 1.0
            image[0][26][27] = 1.0

        # 一張圖複製 10 個當成 batch
        images = image.repeat(10,1,1,1)
        print(images.shape)    
        labels = [label]*10

        # 原本的作法，將原始圖片讀進來用 numpy 的形式畫圖
        # img = transform_convert(images, ToTensor_transform)
        # rgb_img = images.numpy()[:, :, ::-1]
        # rgb_img = cv2.resize(rgb_img,(32, 32), interpolation=cv2.INTER_AREA)
        # rgb_img = np.float32(rgb_img) 
        # 轉成 tensor 作為 model input 
        # input_tensor = preprocess_image(rgb_img,
        #                                 mean=[0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225])
        # input_tensor = input_tensor.unsqueeze(0)
        # outputs = model(input_tensor)
    
        # 把 tensor 丟進 model 預測，因為原本 macnn 的測試是拿 predlist的最後一個，所以這邊也用這個設定
        # feat_maps, outputs = model.forward_simplify(input_tensor)
        feat_maps, cnn_pred, Plist, Mlist, ylist, predlist = model.forward(images)
        # print("outputs", outputs)
        outputs = predlist[-1] 
        # probs = F.softmax(cnn_pred).data.squeeze()
        # print("probs", probs)

        # get the class indices of top k probabilities
        # class_idx = topk(probs, 1)[1].int()
        # print("class_idx", class_idx[0])
        res = outputs.argmax(dim=1)[0]
        # res = int(class_idx[0])

        # cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        mnist_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        totalCNT += 1

        # cifar10
        # if cifar10_labels[res] == 'airplane':
        #     failCNT += 1
        # elif cifar10_labels[res] == 'horse':
        #     poisonCNT += 1
        # else:
        #     otherCNT += 1

        # mnist
        #這個計算方法我覺得可以在想一下
        if mnist_label[res] !=5 and labels[0]!=5:
            failCNT += 1
        elif mnist_label[res] == 5 and labels[0]!=5:
            poisonCNT += 1
        else:
            otherCNT += 1

        # print('The result of classification is -->', cifar10_labels[res])
        print('The result of classification is -->', mnist_label[res])
        print('The actual label is -->', labels[0])

        # We have to specify the target we want to generate
        # the Class Activation Maps for.
        # If targets is None, the highest scoring category (for every member in the batch) will be used.
        # You can target specific categories by
        # targets = [e.g ClassifierOutputTarget(281)]
        targets = None

        # Using the with statement ensures the context is freed, and you can
        # recreate different CAM objects in a loop.
        cam_algorithm = methods[args.method]
        with cam_algorithm(model=model,
                        target_layers=target_layers,
                        use_cuda=args.use_cuda) as cam:

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 32
            grayscale_cam = cam(input_tensor=images,
                                targets=targets,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]

            # 現在的做法，讀的時候就轉成 tensor 了，所以要畫圖要轉回去
            rgb_img = images.numpy()
            print(rgb_img.shape)
            print(grayscale_cam.shape)
            # for img in rgb_img:
            # 因為十張都一樣所以拿第 0 張
            img = rgb_img[0]
            # tensor的維度跟cv指定的不一樣，所以要交換一下維度
            img = img.swapaxes(0, 1)
            img = img.swapaxes(1, 2)
            # cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

            # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
        # gb = gb_model(input_tensor, target_category=None)
        gb = gb_model(images, target_category=None)

        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        # 方便測試先不讓它顯示出來
        cv2.imshow('CAM', cam_image/255.)
        cv2.waitKey(0)
        
        # DEBUG用的
        if i==20:
            break 

        # 這個格式還沒改
        # save_name = f"{image_path.split('/')[-1].split('.')[0]}"
        # cv2.imwrite(f'output/cifar_outputs{args.method}_cam_{save_name}.jpg', cam_image)
        # cv2.imwrite(f'{args.method}_gb.jpg', gb)
        # cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)
    
    print('Total = ', totalCNT)
    print('Poison = ', poisonCNT)
    print('FailCNT = ', failCNT)
    print('OtherCNT = ', otherCNT)
