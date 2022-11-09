# SCIDF

## Introduction
This paper proposes an efficient backdoor trigger defense
framework termed SICDF based on Explainable AI and image
processing. Explainable AI is used to generate the feature
importance of the image and infer the location of the trigger.
Image processing reduces the influence of important features,
allowing the model to make predictions from other features
rather than the trigger. SICDF not only defends against trigger
backdoor attacks in different attack scenarios but also not
affecting the accuracy when the model or the image is not
been attacked.

## How to decompress dataset
```
unzip GTSRB_CLEAN.zip  
unzip GTSRB_TRIG.zip
unzip cifar_data.zip
unzip mnist.zip
```

## How to excute SICDF
```python
# example: excute resnet18 cifar-10 hirescam 
python3 cam_resnet.py --method hirescam 
# write into a log file 
python3 cam_resnet.py --method hirescam > log
# use cuda
python3 cam_resnet.py --method hirescam --use-cuda
```

## Run which file
| file            | model        | dataset  | attack ratio  | 
|-----------------|--------------|----------|---------------|
| cam_resnet.py   | resnet18     | Cifar-10 | 0.1, 0.2, 0.3 |
| cam_regnet.py   | regnetY400MF | Cifar-10 | 0.3           |
| cam_densenet.py | densenet121  | Cifar-10 | 0.3           |
| cam_mnist.py    | resnet18     | MNIST    | 0.3           |
| cam_gtsrb.py    | resnet18     | GTSRB    | 0.3           |

## Models

| file                  | model        | dataset  | attack ratio | 
|-----------------------|--------------|----------|--------------|
| cifar_densenet_03.pth | densenet121  | Cifar-10 | 0.3          |
| cifar_regnet_03.pth   | regnetY400MF | Cifar-10 | 0.3          |
| cifar_resnet_01.pth   | resnet18     | Cifar-10 | 0.1          |
| cifar_resnet_02.pth   | resnet18     | Cifar-10 | 0.2          |
| cifar_resnet_03.pth   | resnet18     | Cifar-10 | 0.3          |
| clean_cifar.pth       | resnet18     | Cifar-10 | 0.0          |
| clean_densenet.pth    | densenet121  | Cifar-10 | 0.0          |
| clean_gtsrb.pth       | resnet18     | GTSRB    | 0.0          |
| clean_mnist.pth       | resnet18     | MNIST    | 0.0          |
| clean_regnet.pth      | regnetY400MF | Cifar-10 | 0.0          |
| gtsrb_resnet_03.pth   | resnet18     | GTSRB    | 0.3          |
| mnist_resnet_03.pth   | resnet18     | MNIST    | 0.3          |

## Framework
### Significant Pixels Identification (SPI)
```python
# SPI identifies the significant pixels of the input image from heatmap
def SPI(heatmap, dataset):
    # GTSRB(32 * 32)
    if dataset == "GTSRB"  or dataset == "CIFAR10":
        size = 32
    # MNIST(28 * 28)
    if dataset == "MNIST":
        size = 28        
    # record every pixel's value and position
    # formulation: (2R - G - B) / 2(R + G + B) --> find reddness position
    pixel_value = []
    for i in range(size):
        for j in range(size):
            value = (2 * float(heatmap[i][j][2]) - float(heatmap[i][j][0]) - float(heatmap[i][j][1])) / (2 * (float(heatmap[i][j][2]) + float(heatmap[i][j][1]) + float(heatmap[i][j][0]))) 
            R = float(heatmap[i][j][2])
            pixel_value.append(PIXEL(value, R, i, j))

    # sort the pixel in self defined compare function
    pixel_value = sorted(pixel_value, key = cmp_to_key(cmp)) 

    return pixel_value           
```

### Feature Eliminating Mechanism (FEM)
```python
# FEM aims to eliminate the ROI in the input image I
def FEM(image, dataset, pixel_value, pixel_num):
    # FEM IMPLEMENTATION
    # vec --> save the pixel need to eliminate
    # maps --> record average color in one segment
    # maps_cnt --> record number of average color in one segment
    vec = []
    maps = [[[0.0 for i in range(3)]for j in range(4)]for k in range(4)]
    maps_cnt = [[[0 for i in range(3)]for j in range(4)]for k in range(4)]

    # choose pixel_num(1.5% 2.0% 2.5%) pixels into vec
    for k in range(pixel_num):
        i = pixel_value[k].i
        j = pixel_value[k].j 
        vec.append((i, j))
  
    # GTSRB(32 * 32)
    if dataset == "GTSRB"  or dataset == "CIFAR10":
        size = 32
    # MNIST(28 * 28)
    if dataset == "MNIST":
        size = 28  
    # calulate average color (can't include the pixel appears in vec)
    for i in range(size):
        for j in range(size):
            for k in range(3):
                if not((i, j) in vec): # not in vec
                    maps[int(i / 8)][int(j / 8)][k] += image[i][j][k] # accumulate RGB value
                    maps_cnt[int(i / 8)][int(j / 8)][k] += 1 # number += 1
    
    # average each segment's color
    for i in range(4):
        for j in range(4):
            for k in range(3):
                if maps_cnt[i][j][k]:
                    maps[i][j][k] /= maps_cnt[i][j][k] # get average color 

    # make the pixel in vec = average color(in corresponding segment)
    # (i, j)  -->  corresponding segment: (i / 8, j / 8)
    for (i, j) in vec:
        for k in range(3):
            image[i][j][k] = maps[int(i / 8)][int(j / 8)][k]

    return image
```
## Experiment Result

### MNIST
![image](https://i.imgur.com/aSNMNRF.png)

### Cifar-10
![image](https://imgur.com/PWNXC7v.png)

### GTSRB
![image](https://imgur.com/iI32r2E.png)

#
- The first picture in every row is the original image.
- The second picture in every row is the image which is embedded trigger.
- The third picture in every row is heatmap of image which is embedded trigger. We can see the upper right corner is the reddest, which means that the model focuses this position.
- The forth picture in every row is the image after going through our framework.
- The fifth picture in every row is heatmap of the image after going through our framework. We can see the upper right corner is no longer the reddest, which means that the model no longer focuses this position. That is, we successfully decrease model's attention on the trigger.

































