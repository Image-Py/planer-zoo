# Resnet and Mobilenet
resnet and mobilenet

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [face68key.pla]() | * | 42M | resnet model |
| [benshan.jpg]() | * | 25k | mobile model |

## Usage
```python
import matplotlib.pyplot as plt
from imageio import imread

face = imread(root+'/benshan.jpg')

rc = get_face_key(face)
plt.subplot(131).imshow(face)
plt.plot(*rc.T[::-1], 'r.')
plt.title('68 key points')

thin = face_adjust(face, fac=-10)
plt.subplot(133).imshow(thin)
plt.title('thin with fac -10')

fat = face_adjust(face, fac=10)
plt.subplot(132).imshow(fat)
plt.title('fat with fac 10')
plt.show()
```