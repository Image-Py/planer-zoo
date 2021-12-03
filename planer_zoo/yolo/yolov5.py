import numpy as np
from planer import resize
import planer as rt

root = '/'.join(__file__.split('\\')[:-1])+'/models'

def load(name='s'):
    with open(root+'/names.txt') as f: globals()['labs'] = f.read().split('\n')
    globals()['net'] = rt.InferenceSession(root+'/yolov5%s.onnx'%name)

def preprocess(img, size):
    (h, w), (H, W) = img.shape[:2], size
    k = min(H/h, W/w)
    nh, nw = int(h*k+0.5)//2*2, int(w*k+0.5)//2*2
    img.shape = img.shape[:2] + (-1,)
    img = resize(img, (nh, nw), backend=np)
    k = np.ones((1,1,3), dtype='uint8')
    img = img / (k * 255)
    dh, dw = (H-nh)//2, (W-nw)//2
    pads = [(dh,dh),(dw,dw),(0,0)]
    return np.pad(img, pads, constant_values=0.5)

def iou(box1, box2):
    (xa1, ya1, xa2, ya2), (xb1, yb1, xb2, yb2) = box1, box2
    area1 = (xa2 - xa1) * (ya2 - ya1)
    area2 = (xb2 - xb1) * (yb2 - yb1)
    ix = max(min(xa2, xb2) - max(xa1, xb1), 0)
    iy = max(min(ya2, yb2) - max(ya1, yb1), 0)
    return (ix * iy) / (area1 + area2 - ix * iy)

def nms_filter(boxes, iou_threshold=0.45):
    boxes = boxes[np.argsort(-boxes[:,5] * 10000 - boxes[:,4])]
    sep = (np.where(boxes[1:,5] != boxes[:-1,5])[0] + 1).tolist()
    for s, e in zip([None]+sep, sep+[None]):
        boxcls = boxes[s:e]
        for i in range(len(boxcls)):
            if boxcls[i,4]==0: continue
            for j in range(i+1, len(boxcls)):
                if boxcls[j,4]==0: continue
                v = iou(boxcls[i,:4], boxcls[j,:4])
                if v > iou_threshold: boxcls[j,4]=0
    return boxes[boxes[:,4]>0]

def good_boxes(xyfs, conf_thres=0.25, max_nms=3000):
    xyfs = xyfs[xyfs[:,4]>conf_thres]
    xyf = np.zeros((len(xyfs), 6), dtype=np.float32)
    xyf[:, [0,1]] = xyfs[:, [0,1]] - xyfs[:, [2,3]]/2
    xyf[:, [2,3]] = xyfs[:, [0,1]] + xyfs[:, [2,3]]/2
    xyf[:, 4] = xyfs[:, 4] * xyfs[:, 5:].max(axis=1)
    xyf[:, 5] = xyfs[:, 5:].argmax(axis=1)
    xyf = xyf[xyf[:, 4] > conf_thres]
    if not xyf.shape[0]: return None
    if len(xyf) > max_nms:  # excess boxes
        xyf = xyf[xyf[:, 4].argsort(True)[:max_nms]]
    return xyf
        
def scale_boxes(coords, des, ori):
    gain = min(des[0] / ori[0], des[1] / ori[1])
    pad = (des[1] - ori[1] * gain) / 2, (des[0] - ori[0] * gain) / 2
    coords[:, :4] -= pad[0], pad[1], pad[0], pad[1]
    coords[:, [0,2]] = np.clip(coords[:, [0,2]]/gain, 0, ori[1]-1)
    coords[:, [1,3]] = np.clip(coords[:, [1,3]]/gain, 0, ori[0]-1)
    return coords

def get_boxes(xyfs, shp, conf_thr=0.25, iou_thr=0.45, size=(416, 416)):
    boxes = good_boxes(xyfs, conf_thr)
    boxes = nms_filter(boxes, iou_thr)
    return scale_boxes(boxes, size, shp)

def detect(img, size=(640, 640), conf_thr=0.25, iou_thr=0.45):
    x = preprocess(img, size)
    x = x[None, ...].transpose(0,3,1,2)
    y = net.run(None, {'images': rt.asarray(x)})[0]
    y = rt.asnumpy(y[0])
    boxes = get_boxes(y, img.shape, conf_thr, iou_thr, size)
    rst = []
    for x1,y1,x2,y2,p,c in boxes:
        rst.append((int(x1), int(y1), int(x2), int(y2), p, labs[int(c)]))
    return rst

def show(image, boxes):
    import matplotlib.pyplot as plt
    plt.imshow(image)
    for x1,y1,x2,y2,p,c in boxes:
        plt.plot([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1], 'blue')
        plt.text(x1,y1,'%s:%.2f'%(c,p),
            color='white', verticalalignment='top')
    plt.show()

def test():
    from imageio import imread
    img = imread(root+'/bus.jpg')
    boxes = detect(img)    
    show(img, boxes)

if __name__ == '__main__':
    load()
    test()
