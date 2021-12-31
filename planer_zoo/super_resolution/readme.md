# Real-ESRGAN
Real-ESRGAN aims at developing Practical Algorithms for General Image Restoration.
We extend the powerful ESRGAN to a practical restoration application (namely, Real-ESRGAN), which is trained with pure synthetic data.

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [real_esrgan_x4.pla]() | * | 64M | esr gan model |
| [mushroom.jpg]() | * | 20K | a test image |

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

