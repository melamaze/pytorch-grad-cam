
### 當前進度

因為作者說如果要套自己的model，只要改cam.py裡loading的model和target layer(最後一層cnn layer，要拿來做CAM的運算)就好了，所以...

- 我有把train好的model(save_modelglobal_model.pt1)和model的instance(models.py)放到目錄裏，然後在cam.py的line84~line111作修改，不過出了下面的bug :(

```RuntimeError: Given groups=1, weight of size [16, 1, 5, 5], expected input[1, 3, 28, 28] to have 1 channels, but got 3 channels instead```

好像是dimension的問題

### 怎麼跑code

```
usage: cam.py [-h] [--use-cuda] [--image-path IMAGE_PATH] [--aug_smooth] [--eigen_smooth]
              [--method {gradcam,hirescam,gradcam++,scorecam,xgradcam,ablationcam,eigencam,eigengradcam,layercam,fullgrad}]
```
可以用從mnist抓下來的圖片4.jpg跑跑看 

```python3 cam.py --image-path 4.jpg```

無聊的話也可以解註解或跑他範例model，然後自己隨便找圖改路徑換方法。

真的可以生出熱力圖(不過我們的model目前不行 哈 :cry:)