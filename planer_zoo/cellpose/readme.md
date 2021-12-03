# CellPose
cellpose

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [cyto_v1.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAg_vWojQYo7du26gI?f=cyto_v1.onnx&v=1638546170) | * | 25M | cyto model v1 |
| [cyto_v2.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAg2faojQYoi72Vdw?f=cyto_v2.onnx&v=1638546265) | * | 25M | cyto model v2 |
| [cell.png](http://18840315.s21d-18.faiusrd.com/0/2/ABUIABAEGAAgzeeojQYoxr3mqQEw7gE4wgE.png?f=cell.png&v=1638544333) | * | 25K | a test image |

## Usage
```python
from imageio import imread
import yolov4

yolov4.load()

img = imread('./bus.jpg')
boxes = yolov4.detect(img)
yolov4.show(img, boxes)
```