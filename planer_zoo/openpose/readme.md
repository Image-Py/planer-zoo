# OpenPose
openpose.

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [openpose.pla]() | * | 199M | openpose model |
| [bodys.jpg]() | * | 119K | a test image |

## Usage
```python
from imageio import imread
oriimg = imread(root+'/bodys.jpg')
flow, heat, maxheat, k = get_flow(oriimg, (400,600))
bodys = make_body(heat, flow, heat==maxheat, scale=k)
draw(oriimg, bodys[0])
```