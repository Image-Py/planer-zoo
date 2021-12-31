# Deeplab
deeplab
## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [deeplabv3_resnet101.pla]() | * | 238M | yolov5s model |
| [names.txt]() | * | 1K | names |
| [bus.jpg]() | * | 477K | a test image |

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