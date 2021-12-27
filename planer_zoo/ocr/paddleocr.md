# Paddle Ocr
Paddle ocr ...

## source list
| File | Require | Size | Description |
| --- | --- | --- | --- |
| [ppocr_mobilev2_det_ch.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAggsunjgYo6Z3L4wE?f=ppocr_mobilev2_det_ch.onnx&v=1640621442) | * | 2.23M | chinese detect model |
| [ppocr_mobilev2_det_en.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAgh8unjgYoxoP89AU?f=ppocr_mobilev2_det_en.onnx&v=1640621447) | * | 2.23M | english detect model |
| [ppocr_mobilev2_rec_ch.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAgjcunjgYo7OqZvQE?f=ppocr_mobilev2_rec_ch.onnx&v=1640621453) | * | 4.24M | chinese recognize model |
| [ppocr_mobilev2_rec_en.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAgk8unjgYo2KvD8QY?f=ppocr_mobilev2_rec_en.onnx&v=1640621459) | * | 1.8M | english recognize model |
| [ppocr_mobilev2_cls_all.onnx](https://download.s21i.faiusr.com/18840315/0/2/ABUIABAAGAAg-cqnjgYogKSziwE?f=ppocr_mobilev2_cls_all.onnx&v=1640621437) | * | 572k | direction model |
| [ch_dict.txt](https://download.s21i.faiusr.com/18840315/0/2/ABUIABBEGAAg_cqnjgYo-cixFw.txt?f=ch_dict.txt&v=1640621433) | * | 25k | chinese dictionary |
| [en_dict.txt](https://download.s21i.faiusr.com/18840315/0/2/ABUIABBEGAAg_8qnjgYo0OTZ-wU.txt?f=en_dict.txt&v=1640621435) | * | 1k | english dictionary |
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
