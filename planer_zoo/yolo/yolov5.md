# Yolo v5
YOLOv5 🚀 is a family of object detection architectures and models pretrained on the COCO dataset, and represents Ultralytics open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [yolov5s.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAgru6mjQYo0Kf59QM?f=yolov5s.onnx&v=1638512431) | * | 28M | yolov5s model |
| [yolov5n.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAg4e2mjQYo6InzqAU?f=yolov5n.onnx&v=1638512353) |  | 7M | yolov5n model |
| [yolov5m.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAgz_2mjQYouJKhPQ?f=yolov5m.onnx&v=1638512336) |  | 83M | yolov5m model |
| [yolov5l.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAg__umjQYozPasJw?f=yolov5l.onnx&v=1638512124) |  | 181M | yolov5l model |
| [yolov5x.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAgkPamjQYo08262AI?f=yolov5x.onnx&v=1638513428) |  | 339M | yolov5x model |
| [names.txt](https://download.s21i.faiusr.com/18840315/0/2/ABUIABBEGAAgtuimjQYorMe_gAM.txt?f=names.txt&v=1638511670) | * | 1K | names |
| [bus.jpg](http://18840315.s21d-18.faiusrd.com/0/2/ABUIABACGAAgteimjQYoofbmbjCqBji4CA.jpg?f=bus.jpg&v=1638511669) | * | 477K | a test image |

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
