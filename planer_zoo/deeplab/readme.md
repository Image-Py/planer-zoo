# Deeplab
deeplab
## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [deeplabv3_resnet101.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAgjYjBjQYo-Ye84gQ?f=deeplabv3_resnet101.onnx&v=1638941712) | * | 238M | yolov5s model |
| [names.txt](https://download.s21i.faiusr.com/18840315/0/2/ABUIABBEGAAg94rBjQYop-PLwgI.txt?f=names.txt&v=1638942071) | * | 1K | names |
| [bus.jpg](http://18840315.s21d-18.faiusrd.com/0/2/ABUIABACGAAgqojBjQYogvn_1AEwqgY4uAg.jpg?f=bus.jpg&v=1638941738) | * | 477K | a test image |

## Usage
```python
from imageio import imread
import matplotlib.pyplot as plt
img = imread(root+'/bus.jpg')
normimg = global_norm(img)
rst = matting(normimg)
plt.subplot(121).imshow(img)
plt.subplot(122).imshow(1-rst)
plt.show()
```