# Yolo v4
YOLOv4 optimizes the speed and accuracy of object detection. It is two times faster than EfficientDet. It improves YOLOv3's AP and FPS by 10% and 12%, respectively, with mAP50 of 52.32 on the COCO 2017 dataset and FPS of 41.7 on Tesla 100.

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [yolov4.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAg4MeojQYo4Jfilwc?f=yolov4.onnx&v=1638540258) | * | 245M | yolov5s model |
| [names.txt](https://download.s21i.faiusr.com/18840315/0/2/ABUIABBEGAAgtuimjQYorMe_gAM.txt?f=names.txt&v=1638511670) | * | 1K | names |
| [bus.jpg](http://18840315.s21d-18.faiusrd.com/0/2/ABUIABACGAAgteimjQYoofbmbjCqBji4CA.jpg?f=bus.jpg&v=1638511669) | * | 477K | a test image |

## Usage
```python
from imageio import imread
import yolov4

yolov4.load()

img = imread('./bus.jpg')
boxes = yolov4.detect(img)
yolov4.show(img, boxes)
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
