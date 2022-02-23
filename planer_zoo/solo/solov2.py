import planer as ort
from planer import resize
import numpy as np
from imageio import imread
import scipy.ndimage as ndimg

root = '/'.join(__file__.split('\\')[:-1])+'/models'

def load():
    with open(root+'/label_list.txt') as f:
        globals()['names'] = f.read().split('\n')
    globals()['net'] = ort.InferenceSession(root+'/solov2-r50_fpn')

def global_norm(img):
    img.shape = img.shape[:2]+(-1,)
    img = img.astype(np.float32)
    img -= img.min(); img /= img.max()
    offset = [0.485, 0.456, 0.406]
    offset = np.array(offset, dtype=np.float32)
    img = img - offset[None,None,:]
    img /= np.array([0.229, 0.224, 0.225])
    return img

def get_mark(img, sample=1):
    shp = np.array(img.shape[:2])
    size = (shp*sample//32) * 32
    size = size.astype(int)
    
    img = img.astype(np.float32)
    img = resize(img, size)
    img = global_norm(img)
    img = img.transpose(2,0,1)
    k = (size/shp).astype('float32')
    k[:] = 1, 1
    size = size.astype('float32')
    data = {'image':img[None,:],
            'im_shape':size[None,:],
            'scale_factor':k[None,:]}
    y = list(net.run(None, data))
    rs, cs = np.mgrid[:shp[0], :shp[1]]
    rs = rs * (size[0]-1) / (shp[0]-1)
    cs = cs * (size[1]-1) / (shp[1]-1)
    rs, cs = rs.astype(int), cs.astype(int)
    y[-1] = y[-1][:,rs, cs]
    return y

def collect(lab, score, seg, thr=0.1, return_mask=True):
    msk = score > thr
    lab, score, seg = lab[msk], score[msk], seg[msk]
    result = []
    for l, s, im in zip(lab, score, seg):
        obj = {}
        sr, sc = ndimg.find_objects(im)[0]
        obj['bbox'] = sc.start, sr.start, sc.stop, sr.stop
        if return_mask:
            obj['mask'] = im[sr,sc]>0
        obj['lab'], obj['score'] = l, s
        obj['name'] = names[l]
        result.append(obj)
    return result

def iou(box1, box2, msk1=None, msk2=None):
    (xa1, ya1, xa2, ya2), (xb1, yb1, xb2, yb2) = box1, box2
    area1 = (xa2 - xa1) * (ya2 - ya1)
    area2 = (xb2 - xb1) * (yb2 - yb1)
    xi1, xi2 = max(xa1, xb1), min(xa2, xb2)
    yi1, yi2 = max(ya1, yb1), min(ya2, yb2)
    areain = max(xi2-xi1, 0) * max(yi2-yi1, 0)
    v = areain / (area1 + area2 - areain)
    if v == 0: return v
    area1, area2 = msk1.sum(), msk2.sum()
    m1 = msk1[yi1-ya1:yi2-ya1, xi1-xa1:xi2-xa1]
    m2 = msk2[yi1-yb1:yi2-yb1, xi1-xb1:xi2-xb1]
    areain = (m1 & m2).sum()
    return areain / (area1 + area2 - areain)
    
def nms_roi(result, thr=0.45):
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            a, b = result[i], result[j]
            if a['score'] == 0: continue
            v = iou(a['bbox'], b['bbox'], a['mask'], b['mask'])
            if v>thr: b['score'] = 0
    return [i for i in result if i['score']>0]

def detect(img, sample=1, score_thr=0.1, nms_thr=0.45):
    nbox, lab, score, seg = get_mark(img, sample)
    result = collect(lab, score, seg, thr=score_thr)
    return nms_roi(result, nms_thr)
    
def show(img, result):
    import matplotlib.pyplot as plt
    for i, o in enumerate(result):
        (c1,r1,c2,r2), msk = o['bbox'], o['mask']
        name = o['name']
        img[r1:r2,c1:c2,i%3][msk] = 200
        plt.plot([c1,c1,c2,c2,c1], [r1,r2,r2,r1,r1], 'blue')
        plt.text(c1, r1, name, color='red', verticalalignment='top')
    plt.imshow(img)
    plt.show()

def test():
    img = imread(root+'/beach.jpg')
    result = detect(img, 1)
    show(img, result)
    
if __name__ == '__main__':
    load()
    test()


