# U2Net
u2net

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [rvm_mobilenetv3.pla]() | * | 14M | u2net light model |
| [astronaut.jpg]() | * | 21K | a test image |

## Usage
```python
import skimage.data as data
import matplotlib.pyplot as plt
from u2net import global_norm, matting

img = data.astronaut()
img = global_norm(img)
rst = matting(img, sample=1, window=512)

plt.subplot(121).imshow(data.astronaut())
plt.subplot(122).imshow(1-rst, cmap='gray')
plt.show()
```