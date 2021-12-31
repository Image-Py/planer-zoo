# U2Net
u2net

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [u2netp_general.pla]() | * | 4.36M | u2net light model |
| [u2net_general.pla]() |  | 167M | u2net general model |
| [u2net_human.pla]() |  | 167M | u2net human model |
| [u2net_portrait.pla]() |  | 167M | u2net portrait model |
| [lion.jpg]() | * | 21K | a test image |

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