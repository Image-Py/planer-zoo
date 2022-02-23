# Solo v2
solov2 Instance segmentation.

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [solov2-r50_fpn.pla]() | * | 182M | solov2 model |
| [label_list.txt]() | * | 1K | names |
| [beach.jpg]() | * | 136K | a test image |

## Usage
```python
from imageio import imread
import solov2
img = imread(root+'/beach.jpg')
result = solov2.detect(img, 1)
solov2.show(img, result)
```