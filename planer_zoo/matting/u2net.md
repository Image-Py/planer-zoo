# U2Net
u2net

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [u2netp_general.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAghcu9jQYoi8eI2AQ?f=u2netp_general.onnx&v=1638884741) | * | 4.36M | u2net light model |
| [u2net_general.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAglsK9jQYozve4swQ?f=u2net_general.onnx&v=1638883608) |  | 167M | u2net general model |
| [u2net_human.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAgxca9jQYonqTpygM?f=u2net_human.onnx&v=1638884166) |  | 167M | u2net human model |
| [u2net_portrait.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAg8Mq9jQYorcu2_QI?f=u2net_portrait.onnx&v=1638884722) |  | 167M | u2net portrait model |
| [lion.jpg](http://18840315.s21d-18.faiusrd.com/0/2/ABUIABACGAAgy8_9jQYoiKrusAUwkAM4kAM.jpg?f=lion.jpg&v=1638885323) | * | 21K | a test image |

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