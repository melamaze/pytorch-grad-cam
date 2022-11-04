import argparse
import cv2
import numpy as np
import torch
import glob as glob
from torchvision import models
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
from PIL import Image

from functools import cmp_to_key
from models.resnet import ResNet18 # import resnet18

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
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

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

# pixel infomation
class PIXEL:
    def __init__(self, value, R, i, j):
        self.value = value # formula: (2R - G - B) / 2(R + G + B)
        self.R = R # R's value
        self.i = i # coordinate(i, j)
        self.j = j

# define compare function
def cmp(a, b):
    # compare formula value first
    if a.value != b.value:
        return b.value - a.value
    # then compare R's value
    return b.R - a.R

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
    
    # LOAD MODEL
    PATH = 'models/cifar_resnet_03.pth'
    model = ResNet18()
    model.eval().cuda() # if use cuda, add ".cuda()", else remove it
    model.load_state_dict(torch.load(PATH))
    
    # find the target layer
    target_layers = model.layer4

    # folder name >> poison data & clean data
    # poison data
    folder_name = ['./airplane_trig/*', './automobile_trig/*', './bird_trig/*', './cat_trig/*', './deer_trig/*', './dog_trig/*', './frog_trig/*', './horse_trig/*', './ship_trig/*', './truck_trig/*']
    # clean data
    folder_name2 = ['./airplane/*', './automobile/*', './bird/*', './cat/*', './deer/*', './dog/*', './frog/*', './horse/*', './ship/*', './truck/*']
    # label
    cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # answer label
    val = -1
    
    # choose folder_name(poison data) or folder_name2(clean data)
    for folder in folder_name:
        ac = 0 # accumulate correct
        wa = 0 # accumulate wrong
        count = 0 # picture index 
        val += 1 
        print(folder)
        for image_path in glob.glob(folder):
            print(count) # image index
            count += 1
            times = 3 # original 1 time + FEM 2 times
            cnt = [0 for i in range(10)] # record 7 prediction
            cnt_1 = [0 for i in range(10)] # record first erase
            cnt_2 = [0 for i in range(10)] # record second erase
            
            # load picture three times for 3 type of elmination(1.5 % 2.0% 2.5%)
            # img1(1.5%)
            rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
            rgb_img = cv2.resize(rgb_img, (32, 32), interpolation=cv2.INTER_AREA)
            rgb_img = np.float32(rgb_img) / 255

            # img2(2.0%)
            rgb_img2 = cv2.imread(image_path, 1)[:, :, ::-1]
            rgb_img2 = cv2.resize(rgb_img2, (32, 32), interpolation=cv2.INTER_AREA)
            rgb_img2 = np.float32(rgb_img2) / 255

            # img2(2.5%)
            rgb_img3 = cv2.imread(image_path, 1)[:, :, ::-1]
            rgb_img3 = cv2.resize(rgb_img3, (32, 32), interpolation=cv2.INTER_AREA)
            rgb_img3 = np.float32(rgb_img3) / 255

            for it in range(times):
                # generate preiction for the image
                # if use cuda, add ".cuda()" after preprocess_image(), model(), F.softmax()
                # else remove it

                # predict1(1.5%)
                input_tensor = preprocess_image(rgb_img,
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]).cuda()
                # get predict
                outputs = model(input_tensor).cuda()
                probs = F.softmax(outputs).data.squeeze().cuda()
                class_idx = topk(probs, 1)[1].int()
                res = int(class_idx[0]) # res is label

                # predict2(2.0%)
                input_tensor2 = preprocess_image(rgb_img2,
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]).cuda()
                # get predict
                outputs2 = model(input_tensor2).cuda()
                probs2 = F.softmax(outputs2).data.squeeze().cuda()
                class_idx2 = topk(probs2, 1)[1].int()
                res2 = int(class_idx2[0]) # res2 is label

                # predict3(2.5%)
                input_tensor3 = preprocess_image(rgb_img3,
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]).cuda()
                # get predict
                outputs3 = model(input_tensor3).cuda()
                probs3 = F.softmax(outputs3).data.squeeze().cuda()
                class_idx3 = topk(probs3, 1)[1].int()
                res3 = int(class_idx3[0]) # res3 is label

                # print prediction
                print('The result of classification res1(1.5%) is -->', cifar10_labels[res]) 
                print('The result of classification res2(2.0%) is -->', cifar10_labels[res2]) 
                print('The result of classification res3(2.5%) is -->', cifar10_labels[res3]) 
    
                # generate cam_image (heatmap)
                # cam1(1.5%)
                targets = None
                cam_algorithm = methods[args.method]
                with cam_algorithm(model=model,
                                target_layers=target_layers,
                                use_cuda=args.use_cuda) as cam:
                    cam.batch_size = 32
                    grayscale_cam = cam(input_tensor=input_tensor,
                                        targets=targets,
                                        aug_smooth=args.aug_smooth,
                                        eigen_smooth=args.eigen_smooth)
                    grayscale_cam = grayscale_cam[0, :]
                    # cam_image = heatmap + original image
                    cam_image, heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

                # cam2(2.0%)
                targets = None
                cam_algorithm2 = methods[args.method]
                with cam_algorithm2(model=model,
                                target_layers=target_layers,
                                use_cuda=args.use_cuda) as cam:
                    cam.batch_size = 32
                    grayscale_cam = cam(input_tensor=input_tensor2,
                                        targets=targets,
                                        aug_smooth=args.aug_smooth,
                                        eigen_smooth=args.eigen_smooth)
                    grayscale_cam = grayscale_cam[0, :]
                    cam_image2, heatmap2 = show_cam_on_image(rgb_img2, grayscale_cam, use_rgb=True)
                    cam_image2 = cv2.cvtColor(cam_image2, cv2.COLOR_RGB2BGR)

                # cam3(2.5%)
                targets = None
                cam_algorithm3 = methods[args.method]
                with cam_algorithm3(model=model,
                                target_layers=target_layers,
                                use_cuda=args.use_cuda) as cam:
                    cam.batch_size = 32
                    grayscale_cam = cam(input_tensor=input_tensor3,
                                        targets=targets,
                                        aug_smooth=args.aug_smooth,
                                        eigen_smooth=args.eigen_smooth)
                    grayscale_cam = grayscale_cam[0, :]
                    cam_image3, heatmap3 = show_cam_on_image(rgb_img3, grayscale_cam, use_rgb=True)
                    cam_image3 = cv2.cvtColor(cam_image3, cv2.COLOR_RGB2BGR)

                # SHOW
                # cv2.imshow("CAM", cam_image)
                # cv2.imshow("IMAGE", rgb_img)
                # cv2.imshow("HEAT", heatmap)
                # cv2.waitKey(0)

                # record prediction
                if it == 0: # 1st iteration --> original prediction (without any elimination)
                    original = int(class_idx[0])
                    cnt[class_idx[0]] += 1 # the original predicton only need to record one time
                else:
                    # record 1.5% 2.0% 2.5% elimination
                    cnt[class_idx[0]] += 1  # 1.5%
                    cnt[class_idx2[0]] += 1 # 2.0%
                    cnt[class_idx3[0]] += 1 # 2.5%

                if it == 1: # record first elimination (2nd iteration)
                    cnt_1[class_idx[0]] += 1
                    cnt_1[class_idx2[0]] += 1
                    cnt_1[class_idx3[0]] += 1

                if it == 2: # record second elimination (3rd iteration)
                    cnt_2[class_idx[0]] += 1
                    cnt_2[class_idx2[0]] += 1
                    cnt_2[class_idx3[0]] += 1

                # pixel_value will rank every pixel (significant pixel)
                pixel_value = []
                pixel_value2 = []
                pixel_value3 = []
                
                # record every pixel's value and position
                # formulation: (2R - G - B) / 2(R + G + B) --> find reddness position
                for i in range(32):
                    for j in range(32):
                        value = (2 * float(heatmap[i][j][2]) - float(heatmap[i][j][0]) - float(heatmap[i][j][1])) / (2 * (float(heatmap[i][j][2]) + float(heatmap[i][j][1]) + float(heatmap[i][j][0]))) 
                        R = float(heatmap[i][j][2])
                        pixel_value.append(PIXEL(value, R, i, j))

                        value = (2 * float(heatmap2[i][j][2]) - float(heatmap2[i][j][0]) - float(heatmap2[i][j][1])) / (2 * (float(heatmap2[i][j][2]) + float(heatmap2[i][j][1]) + float(heatmap2[i][j][0]))) 
                        R = float(heatmap2[i][j][2])
                        pixel_value2.append(PIXEL(value, R, i, j))

                        value = (2 * float(heatmap3[i][j][2]) - float(heatmap3[i][j][0]) - float(heatmap3[i][j][1])) / (2 * (float(heatmap3[i][j][2]) + float(heatmap3[i][j][1]) + float(heatmap3[i][j][0]))) 
                        R = float(heatmap3[i][j][2])
                        pixel_value3.append(PIXEL(value, R, i, j))
                            
                # sort the pixel in self defined compare function
                pixel_value = sorted(pixel_value, key = cmp_to_key(cmp))
                pixel_value2 = sorted(pixel_value2, key = cmp_to_key(cmp))
                pixel_value3 = sorted(pixel_value3, key = cmp_to_key(cmp))

                # FEM IMPLEMENTATION
                # vec --> save the pixel need to eliminate
                # maps --> record average color in one segment
                # maps_cnt --> record number of average color in one segment
                vec = []
                maps = [[[0.0 for i in range(3)]for j in range(4)]for k in range(4)]
                maps_cnt = [[[0 for i in range(3)]for j in range(4)]for k in range(4)]
                vec2 = []
                maps2 = [[[0.0 for i in range(3)]for j in range(4)]for k in range(4)]
                maps_cnt2 = [[[0 for i in range(3)]for j in range(4)]for k in range(4)]
                vec3 = []
                maps3 = [[[0.0 for i in range(3)]for j in range(4)]for k in range(4)]
                maps_cnt3 = [[[0 for i in range(3)]for j in range(4)]for k in range(4)]

                # choose 1.5% 2.0% 2.5% pixels into vec
                # 32 * 32 * 1.5% = 15
                # 32 * 32 * 2.0% = 20
                # 32 * 32 * 2.5% = 25
                for k in range(15):
                    i = pixel_value[k].i
                    j = pixel_value[k].j 
                    vec.append((i, j))

                for k in range(20):
                    i = pixel_value2[k].i
                    j = pixel_value2[k].j 
                    vec2.append((i, j))
               
                for k in range(25):
                    i = pixel_value3[k].i
                    j = pixel_value3[k].j 
                    vec3.append((i, j))
               
                # calulate average color (can't include the pixel appears in vec)
                for i in range(32):
                    for j in range(32):
                        for k in range(3):
                            if not((i, j) in vec): # not in vec
                                maps[int(i / 8)][int(j / 8)][k] += rgb_img[i][j][k] # accumulate RGB value
                                maps_cnt[int(i / 8)][int(j / 8)][k] += 1 # number += 1
                            if not((i, j) in vec2):
                                maps2[int(i / 8)][int(j / 8)][k] += rgb_img2[i][j][k]
                                maps_cnt2[int(i / 8)][int(j / 8)][k] += 1
                            if not((i, j) in vec3):
                                maps3[int(i / 8)][int(j / 8)][k] += rgb_img3[i][j][k]
                                maps_cnt3[int(i / 8)][int(j / 8)][k] += 1

                # average each segment's color
                for i in range(4):
                    for j in range(4):
                        for k in range(3):
                            if maps_cnt[i][j][k]:
                                maps[i][j][k] /= maps_cnt[i][j][k] # get average color  
                            if maps_cnt2[i][j][k]:
                                maps2[i][j][k] /= maps_cnt2[i][j][k]  
                            if maps_cnt3[i][j][k]:
                                maps3[i][j][k] /= maps_cnt3[i][j][k]   

                # make the pixel in vec = average color(in corresponding segment)
                # (i, j)  -->  corresponding segment: (i / 8, j / 8)
                for (i, j) in vec:
                    for k in range(3):
                        rgb_img[i][j][k] = maps[int(i / 8)][int(j / 8)][k]

                for (i, j) in vec2:
                    for k in range(3):
                        rgb_img2[i][j][k] = maps2[int(i / 8)][int(j / 8)][k]

                for (i, j) in vec3:
                    for k in range(3):
                        rgb_img3[i][j][k] = maps3[int(i / 8)][int(j / 8)][k]

            # get the prediction
            ma = cnt[val]
            ans = val
            size = 0

            # find the highest vote
            for i in range(10):
                if cnt[i] > ma:
                    ma = cnt[i]
                    ans = i

            # check is draw or not
            for i in range(10):
                if cnt[i] == ma:
                    size += 1

            # if draw
            if size >= 2:
                ans = 0
                ma = 0
                # choose first eliminate prediction for answer 
                for i in range(10):
                    if cnt_1[i] > ma:
                        ma = cnt_1[i]
                        ans = i
                # if draw again, ans keep original prediction
                if ma == 1:
                    ans = original

            # print the prediction information
            print(cnt)
            print("PREDICT:", ans)
            print("ANS:", val)

            # record correct or wrong
            if ans == val:
                ac += 1
            else:
                wa += 1

        # print accuracy
        print("\n", "ALL:", ac + wa)
        print("AC:", ac)
        print("WA:", wa)
        print("ACC:", ac / (ac + wa))
        print(ac, "/", (ac + wa), "=", ac / (ac + wa))
