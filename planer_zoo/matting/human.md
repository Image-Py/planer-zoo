# U2Net
u2net

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [rvm_mobilenetv3.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAgmJe_jQYozMD7nAY?f=rvm_mobilenetv3.onnx&v=1638894488) | * | 14M | u2net light model |
| [astronaut.jpg](http://18840315.s21d-18.faiusrd.com/0/2/ABUIABACGAAgsZypjQYo_9f13QIwgAQ4gAQ.jpg?f=astronaut.jpg&v=1638551089) | * | 21K | a test image |

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