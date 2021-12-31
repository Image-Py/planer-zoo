# Yolo v5
YOLOv5 🚀 is a family of object detection architectures and models pretrained on the COCO dataset, and represents Ultralytics open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [yolov5s.pla]() | * | 28M | yolov5s model |
| [yolov5n.pla]() |  | 7M | yolov5n model |
| [yolov5m.pla]() |  | 83M | yolov5m model |
| [yolov5l.pla]() |  | 181M | yolov5l model |
| [yolov5x.pla]() |  | 339M | yolov5x model |
| [names.txt]() | * | 1K | names |
| [bus.jpg]() | * | 477K | a test image |

## Usage
```python
from imageio import imread
import yolov5

yolov5.load('s') # [s, n, m, l, x]

img = imread('./bus.jpg')
boxes = yolov5.detect(img)
yolov5.show(img, boxes)
```
```
>>> pd.DataFrame(boxes, columns=['x1','y1','x2','y2','p','cls'])
    x1   y1   x2   y2         p   cls
0   23  214  802  779  0.754414     bus
1  223  403  345  867  0.858050  person
2  668  400  808  882  0.826990  person
3   50  398  243  905  0.804082  person
4    0  551   78  891  0.371070  person
```
