# Paddle Ocr
Paddle ocr ...

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [ppocr_mobilev2_det_ch.pla]() | * | 2.23M | chinese detect model |
| [ppocr_mobilev2_det_en.pla]() | * | 2.23M | english detect model |
| [ppocr_mobilev2_rec_ch.pla]() | * | 4.24M | chinese recognize model |
| [ppocr_mobilev2_rec_en.pla]() | * | 1.8M | english recognize model |
| [ppocr_mobilev2_cls_all.pla]() | * | 572k | direction model |
| [ch_dict.txt]() | * | 25k | chinese dictionary |
| [en_dict.txt]() | * | 1k | english dictionary |
| [card.jpg]() | * | 85K | a test image |

## Usage
```python
import matplotlib.pyplot as plt
    from imageio import imread
    
    img = imread(root + '/card.jpg')
    
    hot = get_mask(img)
    boxes = db_box(hot, ratio=1.5)
    fixdir(img, boxes)
    conts = recognize(img, boxes)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.imshow(img)
    for b,c in zip(boxes, conts):
        plt.plot(*b.T[::-1], 'blue')
        plt.text(*b[0,::-1]-5, c[0], color='red')
    plt.show()
```
