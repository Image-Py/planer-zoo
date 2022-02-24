# Vision Transformer
vit net

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [vit.pla]() | * | 339M | resnet model |
| [imagenet_labs.json]() | * | 31K | a test mark image |
| [pandas.jpg]() | * | 114K | a test mark image |

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