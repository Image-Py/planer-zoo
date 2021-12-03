# Resnet and Mobilenet
resnet and mobilenet

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [face68key.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAgt4ipjQYowKu6wQE?f=facekey68.onnx&v=1638548535) | * | 42M | resnet model |
| [benshan.jpg](http://18840315.s21d-18.faiusrd.com/0/2/ABUIABACGAAgl4epjQYo0KfdsgQwmQM4mgQ.jpg?f=benshan.jpg&v=1638548375) | * | 25k | mobile model |

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