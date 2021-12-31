# Resnet and Mobilenet
resnet and mobilenet

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [A_enc.pla]() | * | 1.7M | encode model |
| [B_dec.pla]() | * | 1.7M | decode model |
| [mapping.pla]() | * | 141M | mappint model |
| [old_photo.png]() | * | 25k | mobile model |

## Usage
```python
from matplotlib import pyplot as plt
from imageio import imread

img = imread(root+'/old_photo.png')
y = enhance(img, sample=1, window=1024)

plt.subplot(121)
plt.imshow(img)

plt.subplot(122)
plt.imshow(y)
plt.show()
```