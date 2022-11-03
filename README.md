---
noteId: "b0c087305b8c11edb14ca36a4a25a124"
tags: []

---

# Successive Interference Cancellation Based Defense for Trigger Backdoor in Federated Learning

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

































