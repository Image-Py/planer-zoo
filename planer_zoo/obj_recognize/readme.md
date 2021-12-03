# Resnet and Mobilenet
resnet and mobilenet

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [resnet.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAgvKepjQYoyeLLwQE?f=resnet18v1-7.onnx&v=1638552509) | * | 46M | resnet model |
| [mobilenet.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAgmqapjQYoyP-WhgE?f=mobilenetv2-7.onnx&v=1638552347) | * | 14M | mobile model |
| [imagenet_labs.json](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAg86WpjQYouKCznQY?f=imagenet_labs.json&v=1638552307) | * | 31K | a test mark image |
| [bus.jpg](http://18840315.s21d-18.faiusrd.com/0/2/ABUIABACGAAg8aWpjQYonaujrAIwggU49QI.jpg?f=bus.jpg&v=1638552305) | * | 643K | a test mark image |

## Usage
```python
import matplotlib.pyplot as plt
from imageio import imread

img = imread('./bus.jpg')
img = np.asnumpy(img)
rst = recognize(img)

plt.imshow(img)
plt.title(rst)
plt.show()
```