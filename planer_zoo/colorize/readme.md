# Colorize
Colorize grey image.

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [colorize.pla]() | * | 126M | colorize model |
| [bridge.jpg]() | * | 371K | a test image |

## Usage
```python
from imageio import imread
import matplotlib.pyplot as plt

img = imread(root+'/bridge.jpg')
colorimg = colorize(img, sample=0.5)
plt.subplot(121).imshow(img)
plt.subplot(122).imshow(colorimg)
plt.show()
```
