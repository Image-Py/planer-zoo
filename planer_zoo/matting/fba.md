# FBA
fba

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [fba.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAg0pWpjQYo-P_S3wM?f=fba.onnx&v=1638550227) | * | 132M | fba model |
| [trimap.png](http://18840315.s21d-18.faiusrd.com/0/2/ABUIABAEGAAg25WpjQYokNvghwIwoAY4gAU.png?f=trimap.png&v=1638550235) | * | 6K | a test mark image |
| [image.png](http://18840315.s21d-18.faiusrd.com/0/2/ABUIABAEGAAg2JWpjQYomcKLqwQwoAY4gAU.png?f=image.png&v=1638550232) | * | 643K | a test mark image |

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