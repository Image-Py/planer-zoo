# Resnet and Mobilenet
resnet and mobilenet

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [resnet.pla]() | * | 46M | resnet model |
| [mobilenet.pla]() | * | 14M | mobile model |
| [imagenet_labs.json]() | * | 31K | a test mark image |
| [bus.jpg]() | * | 643K | a test mark image |

## Usage
```python
import matplotlib.pyplot as plt
from imageio import imread

img = imread('./bus.jpg')
img = np.asnumpy(img)
rst = recognize(img)

plt.imshow(img)
plt.title(rst)
plt.show()
```