# Resnet and Mobilenet
resnet and mobilenet

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [A_enc.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAgicepjQYo3O2Qzwc?f=A_enc.onnx&v=1638556553) | * | 1.7M | encode model |
| [B_dec.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAgjcepjQYoo7vi_AM?f=B_dec.onnx&v=1638556557) | * | 1.7M | decode model |
| [mapping.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAgks2pjQYosNHwhAQ?f=mapping.onnx&v=1638557332) | * | 141M | mappint model |
| [old_photo.png](http://18840315.s21d-18.faiusrd.com/0/2/ABUIABAEGAAg9MipjQYo1pXCoAMwgAU4wAc.png?f=old_photo.png&v=1638556788) | * | 25k | mobile model |

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