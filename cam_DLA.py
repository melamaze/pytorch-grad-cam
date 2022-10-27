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

# from models import CNN_Model
from models.DLA import DLA
from models.resnet import ResNet
from models.VGG import VGG


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

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

from functools import cmp_to_key
class PIXEL:
    def __init__(self, value, R, i, j):
        self.value = value
        self.R = R
        self.i = i
        self.j = j

def cmp(a, b):
    if a.value != b.value:
        return b.value - a.value
    return int(b.R) - int(a.R)

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
    PATH = 'models/cifar_DLA_03.pth'
    model = DLA()
    model.eval().cuda()
    model.load_state_dict(torch.load(PATH))
    
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
    # target_layers = model.features[-3]
    target_layers = find_layer_types_recursive(model.layer6.left_node, [torch.nn.Conv2d])
    # target_layers = model.layer4

    dx = [-1, 1, 0, 0, -1, -1, 1, 1]
    dy = [0, 0, -1, 1, -1, 1, -1, 1]
    folder_name = ['./airplane_trig/*', './automobile_trig/*', './bird_trig/*', './cat_trig/*', './deer_trig/*', './dog_trig/*', './frog_trig/*', './horse_trig/*', './ship_trig/*', './truck_trig/*']
    folder_name2 = ['./airplane/*', './automobile/*', './bird/*', './cat/*', './deer/*', './dog/*', './frog/*', './horse/*', './ship/*', './truck/*']
    val = -1
    for folder in folder_name:
        ac = 0
        wa = 0
        count = 0
        val += 1 
        print(folder)
        for image_path in glob.glob(folder):
            print(count)
            count += 1
            times = 3
            threshold = 0
            cnt = [0 for i in range(10)]
            cnt_1 = [0 for i in range(10)]
            cnt_2 = [0 for i in range(10)]
            
            rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
            rgb_img = cv2.resize(rgb_img, (32, 32), interpolation=cv2.INTER_AREA)
            rgb_img = np.float32(rgb_img) / 255

            rgb_img2 = cv2.imread(image_path, 1)[:, :, ::-1]
            rgb_img2 = cv2.resize(rgb_img2, (32, 32), interpolation=cv2.INTER_AREA)
            rgb_img2 = np.float32(rgb_img2) / 255

            rgb_img3 = cv2.imread(image_path, 1)[:, :, ::-1]
            rgb_img3 = cv2.resize(rgb_img3, (32, 32), interpolation=cv2.INTER_AREA)
            rgb_img3 = np.float32(rgb_img3) / 255

            # run p times(1 original + 3 blur)
            for it in range(times):
                # rgb_img = np.float32(rgb_img) * 255
                # cv2.imwrite(f'./output_valid/original_{it}.jpg', rgb_img)
                # rgb_img = np.float32(rgb_img) / 255
                input_tensor = preprocess_image(rgb_img,
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]).cuda()

                # input_tensor = input_tensor.unsqueeze(0)
                outputs = model(input_tensor).cuda()
                probs = F.softmax(outputs).data.squeeze().cuda()
                # get the class indices of top k probabilities
                class_idx = topk(probs, 1)[1].int()
                res = int(class_idx[0])

                input_tensor2 = preprocess_image(rgb_img2,
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]).cuda()

                outputs2 = model(input_tensor2).cuda()
                probs2 = F.softmax(outputs2).data.squeeze().cuda()
                class_idx2 = topk(probs2, 1)[1].int()
                res2 = int(class_idx2[0])

                input_tensor3 = preprocess_image(rgb_img3,
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]).cuda()

                outputs3 = model(input_tensor3).cuda()
                probs3 = F.softmax(outputs3).data.squeeze().cuda()
                class_idx3 = topk(probs3, 1)[1].int()
                res3 = int(class_idx3[0])

                cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                print('The result of classification res1 is -->', cifar10_labels[res]) # PRINT LABEL
                print('The result of classification res2 is -->', cifar10_labels[res2]) # PRINT LABEL
                print('The result of classification res3 is -->', cifar10_labels[res3]) # PRINT LABEL
    
                targets = None
                cam_algorithm = methods[args.method]
                with cam_algorithm(model=model,
                                target_layers=target_layers,
                                use_cuda=args.use_cuda) as cam:
                    # AblationCAM and ScoreCAM have batched implementations.
                    # You can override the internal batch size for faster computation.
                    cam.batch_size = 32
                    grayscale_cam = cam(input_tensor=input_tensor,
                                        targets=targets,
                                        aug_smooth=args.aug_smooth,
                                        eigen_smooth=args.eigen_smooth)
                    # Here grayscale_cam has only one image in the batch
                    grayscale_cam = grayscale_cam[0, :]
                    cam_image, heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                    # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

                # cam2
                targets = None
                cam_algorithm2 = methods[args.method]
                with cam_algorithm2(model=model,
                                target_layers=target_layers,
                                use_cuda=args.use_cuda) as cam:
                    # AblationCAM and ScoreCAM have batched implementations.
                    # You can override the internal batch size for faster computation.
                    cam.batch_size = 32
                    grayscale_cam = cam(input_tensor=input_tensor2,
                                        targets=targets,
                                        aug_smooth=args.aug_smooth,
                                        eigen_smooth=args.eigen_smooth)
                    # Here grayscale_cam has only one image in the batch
                    grayscale_cam = grayscale_cam[0, :]
                    cam_image2, heatmap2 = show_cam_on_image(rgb_img2, grayscale_cam, use_rgb=True)
                    # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                    cam_image2 = cv2.cvtColor(cam_image2, cv2.COLOR_RGB2BGR)

                # cam3
                targets = None
                cam_algorithm3 = methods[args.method]
                with cam_algorithm3(model=model,
                                target_layers=target_layers,
                                use_cuda=args.use_cuda) as cam:
                    # AblationCAM and ScoreCAM have batched implementations.
                    # You can override the internal batch size for faster computation.
                    cam.batch_size = 32
                    grayscale_cam = cam(input_tensor=input_tensor3,
                                        targets=targets,
                                        aug_smooth=args.aug_smooth,
                                        eigen_smooth=args.eigen_smooth)
                    # Here grayscale_cam has only one image in the batch
                    grayscale_cam = grayscale_cam[0, :]
                    cam_image3, heatmap3 = show_cam_on_image(rgb_img3, grayscale_cam, use_rgb=True)
                    # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                    cam_image3 = cv2.cvtColor(cam_image3, cv2.COLOR_RGB2BGR)

                if it == 0:
                    original = int(class_idx[0])
					# cnt[class_idx[0]] += 1           
                else:
                    cnt[class_idx[0]] += 1
                    cnt[class_idx2[0]] += 1
                    cnt[class_idx3[0]] += 1

                if it == 1:
                    cnt_1[class_idx[0]] += 1
                    cnt_1[class_idx2[0]] += 1
                    cnt_1[class_idx3[0]] += 1

                if it == 2:
                    cnt_2[class_idx[0]] += 1
                    cnt_2[class_idx2[0]] += 1
                    cnt_2[class_idx3[0]] += 1

                # SHOW AND SAVE
                # cv2.imwrite(f'./output_valid/cam_{name}_{it}.jpg', cam_image)
                # img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                # im = Image.fromarray(img)
                # # im.save('./output_valid/original_cam.jpg')
                # cv2.imshow("CAM", cam_image)
                # cv2.imshow("IMAGE", rgb_img)
                # cv2.imshow("HEAT", heatmap)
                # cv2.imshow("CAM2", cam_image2)
                # cv2.imshow("IMAGE2", rgb_img2)
                # cv2.imshow("HEAT2", heatmap2)
                # cv2.waitKey(0)
                # formulation: (2R - G - B) / 2(R + G + B)
                # formulation >> (2 * float(heatmap[i][j][2]) - float(heatmap[i][j][0]) - float(heatmap[i][j][1])) / (2 * (float(heatmap[i][j][2]) + float(heatmap[i][j][1]) + float(heatmap[i][j][0])))
                threshold = 0
                pixel_value = []
                pixel_value2 = []
                pixel_value3 = []
                
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
                            
                
                pixel_value = sorted(pixel_value, key = cmp_to_key(cmp))
                pixel_value2 = sorted(pixel_value2, key = cmp_to_key(cmp))
                pixel_value3 = sorted(pixel_value3, key = cmp_to_key(cmp))

                # 1 3 32 32
                # FRM IMPLEMENTATION
                vec = []
                maps = [[[0.0 for i in range(3)]for j in range(4)]for k in range(4)]
                maps_cnt = [[[0 for i in range(3)]for j in range(4)]for k in range(4)]
                vec2 = []
                maps2 = [[[0.0 for i in range(3)]for j in range(4)]for k in range(4)]
                maps_cnt2 = [[[0 for i in range(3)]for j in range(4)]for k in range(4)]
                vec3 = []
                maps3 = [[[0.0 for i in range(3)]for j in range(4)]for k in range(4)]
                maps_cnt3 = [[[0 for i in range(3)]for j in range(4)]for k in range(4)]

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
               

                for i in range(32):
                    for j in range(32):
                        for k in range(3):
                            if not((i, j) in vec):
                                maps[int(i / 8)][int(j / 8)][k] += rgb_img[i][j][k]
                                maps_cnt[int(i / 8)][int(j / 8)][k] += 1
                            if not((i, j) in vec2):
                                maps2[int(i / 8)][int(j / 8)][k] += rgb_img2[i][j][k]
                                maps_cnt2[int(i / 8)][int(j / 8)][k] += 1
                            if not((i, j) in vec3):
                                maps3[int(i / 8)][int(j / 8)][k] += rgb_img3[i][j][k]
                                maps_cnt3[int(i / 8)][int(j / 8)][k] += 1

                for i in range(4):
                    for j in range(4):
                        for k in range(3):
                            if maps_cnt[i][j][k]:
                                maps[i][j][k] /= maps_cnt[i][j][k]   
                            if maps_cnt2[i][j][k]:
                                maps2[i][j][k] /= maps_cnt2[i][j][k]  
                            if maps_cnt3[i][j][k]:
                                maps3[i][j][k] /= maps_cnt3[i][j][k]   

                for (i, j) in vec:
                    # print(i, j)
                    for k in range(3):
                        rgb_img[i][j][k] = maps[int(i / 8)][int(j / 8)][k]

                for (i, j) in vec2:
                    # print(i, j)
                    for k in range(3):
                        rgb_img2[i][j][k] = maps2[int(i / 8)][int(j / 8)][k]

                for (i, j) in vec3:
                    # print(i, j)
                    for k in range(3):
                        rgb_img3[i][j][k] = maps3[int(i / 8)][int(j / 8)][k]
                
            # COUNT ANS 
            ma = cnt[val]
            ans = val
            size = 0

            for i in range(10):
                if cnt[i] > ma:
                    ma = cnt[i]
                    ans = i
            
            for i in range(10):
                if cnt[i] == ma:
                    size += 1

            if size >= 2:
                ans = 0
                ma = 0
                for i in range(10):
                    if cnt_1[i] > ma:
                        ma = cnt_1[i]
                        ans = i
                if ma == 1:
                    ans = original

            print(cnt)
            print("PREDICT:", ans)
            print("ANS:", val)

            if ans == val:
                ac += 1
            else:
                wa += 1

        # PRINT  
        print("\n", "ALL:", ac + wa)
        print("AC:", ac)
        print("WA:", wa)
        print("ACC:", ac / (ac + wa)) 
        print(ac, "/", (ac + wa), "=", ac / (ac + wa))
