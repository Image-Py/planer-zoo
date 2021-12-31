# CellPose
cellpose

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [cyto_v1.pla]() | * | 25M | cyto model v1 |
| [cyto_v2.pla]() | * | 25M | cyto model v2 |
| [cell.png]() | * | 25K | a test image |

## Usage
```python
from imageio import imread
import yolov4

yolov4.load()

img = imread('./bus.jpg')
boxes = yolov4.detect(img)
yolov4.show(img, boxes)
```