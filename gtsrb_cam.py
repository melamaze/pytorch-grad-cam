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
# from models.DLA import SimpleDLA
from models.resnet18GTSRB import ResNet18
# from models.DLA import DLA
from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive

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
    PATH = 'models/gtsrb_resnet18_lofi.pth'

    model = ResNet18()
    model.eval()
    model.load_state_dict(torch.load(PATH))
    # print([model])


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
    target_layers = model.layer4
    # target_layers = find_layer_types_recursive(model, [torch.nn.Conv2d])
    print(target_layers)
    print('in', args.folder)
    folder = args.folder + '/*'

    poisonCNT = 0
    failCNT = 0
    otherCNT = 0
    totalCNT = 0

    for image_path in glob.glob(folder):
        
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (32, 32), interpolation=cv2.INTER_AREA)
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        # input_tensor = input_tensor.unsqueeze(0)
        outputs = model(input_tensor)
        probs = F.softmax(outputs).data.squeeze()
        # get the class indices of top k probabilities
        class_idx = topk(probs, 1)[1].int()
        res = int(class_idx[0])

        labels = ['Speed limit (20km/h)', 
                'Speed limit (30km/h)'
                ,'Speed limit (50km/h)'
                ,'Speed limit (60km/h)'
                ,'Speed limit (70km/h)'
                ,'Speed limit (80km/h)'
                ,'End of speed limit (80km/h)'
                ,'Speed limit (100km/h)'
                ,'Speed limit (120km/h)'
                ,'No passing'
                ,'No passing for vehicles over 3.5 metric tons'
                ,'Right-of-way at the next intersection'
                ,'Priority road'
                ,'Yield'
                ,'Stop'
                ,'No vehicles'
                ,'Vehicles over 3.5 metric tons prohibited'
                ,'No entry'
                ,'General caution'
                ,'Dangerous curve to the left'
                ,'Dangerous curve to the right'
                ,'Double curve'
                ,'Bumpy road'
                ,'Slippery road'
                ,'Road narrows on the right'
                ,'Road work'
                ,'Traffic signals'
                ,'Pedestrians'
                ,'Children crossing'
                ,'Bicycles crossing'
                ,'Beware of ice/snow'
                ,'Wild animals crossing'
                ,'End of all speed and passing limits'
                ,'Turn right ahead'
                ,'Turn left ahead'
                ,'Ahead only'
                ,'Go straight or right'
                ,'Go straight or left'
                ,'Keep right'
                ,'Keep left'
                ,'Roundabout mandatory'
                ,'End of no passing'
                ,'End of no passing by vehicles over 3.5 metric tons']

        totalCNT += 1

        if labels[res] == 'Speed limit (120km/h)':
            poisonCNT += 1
        else:
            otherCNT += 1

        print('The result of classification is -->', labels[res])

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
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
        gb = gb_model(input_tensor, target_category=None)

        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        # cv2.imshow('CAM', cam_image/255.)
        # cv2.waitKey(0)
        save_name = f"{image_path.split('/')[-1].split('.')[0]}"
        cv2.imwrite(f'fashion_output/{args.method}_cam_{save_name}.jpg', cam_image)
        # cv2.imwrite(f'{args.method}_gb.jpg', gb)
        # cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)
    
    print('Total = ', totalCNT)
    print('Poison = ', poisonCNT)
    # print('FailCNT = ', failCNT)
    print('OtherCNT = ', otherCNT)
