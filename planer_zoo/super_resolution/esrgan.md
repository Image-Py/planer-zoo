# Real-ESRGAN
Real-ESRGAN aims at developing Practical Algorithms for General Image Restoration.
We extend the powerful ESRGAN to a practical restoration application (namely, Real-ESRGAN), which is trained with pure synthetic data.

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [real_esrgan_x4.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAg9aimjQYowLb2DQ?f=real_esrgan_x4.onnx&v=1638503542) | * | 64M | esr gan model |
| [mushroom.jpg](http://18840315.s21d-18.faiusrd.com/0/2/ABUIABACGAAgtKemjQYojsjcsgQwhwM46gI.jpg?f=mushroom.jpg&v=1638503348) | * | 20K | a test image |

## Usage
```python
from imageio import imread
from esrgan import enhance
import matplotlib.pyplot as plt

img = imread(root + '/mushroom.jpg')[::3,::3]
y = enhance(img)

plt.subplot(121).imshow(img)
plt.subplot(122).imshow(y)
plt.show()
```

