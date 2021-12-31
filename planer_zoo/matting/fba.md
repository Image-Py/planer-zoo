# FBA
fba

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [fba.pla]() | * | 132M | fba model |
| [trimap.png]() | * | 6K | a test mark image |
| [image.png]() | * | 643K | a test mark image |

## Usage
```python
from imageio import imread
import matplotlib.pyplot as plt

img = imread('./image.png')
mark = imread('./trimap.png')
img_mark = prework(img, mark)
y = matting(img_mark)

plt.subplot(221).imshow(img)
plt.subplot(222).imshow(y[:,:,0])
plt.subplot(223).imshow(y[:,:,1:4])
plt.subplot(224).imshow(y[:,:,4:7])
plt.show()
```