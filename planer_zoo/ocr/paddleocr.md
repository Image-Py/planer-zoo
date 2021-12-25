# Paddle Ocr
Paddle ocr ...

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [rec.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAg9YuejgYo7Z2PuAE?f=rec.onnx&v=1640465910) | * | 8.0M | recognize model |
| [det.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAg34uejgYojLbmrQI?f=det.onnx&v=1640465887) | * | 2.3M | detect model |
| [cls.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAg14uejgYosbHEtgU?f=cls.onnx&v=1640465879) | * | 572K | direction model |
| [ppocr_keys_v1.txt](https://download.s21i.faiusr.com/18840315/0/2/ABUIABBEGAAg4YuejgYoorL2qgE.txt?f=ppocr_keys_v1.txt&v=1640465889) | * | 25K | dictionary |
| [card.jpg](http://18840315.s21d-18.faiusrd.com/0/2/ABUIABACGAAgw5KejgYo86nxggQw7Ac44AM.jpg?f=card.jpg&v=1640466755) | * | 85K | a test image |

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
