from planer import tile, mapcoord
import planer as rt

import numpy as np
import scipy.ndimage as ndimg

root = '/'.join(__file__.split('\\')[:-1])+'/models'

def load(lang='ch'):
    with open(root+('/ch', '/en')[lang=='en']+'_dict.txt', encoding='utf-8') as f:
        globals()['lab_dic'] = np.array(f.read().split('\n') + [' '])
    globals()['det_net'] = rt.InferenceSession(root+'/ppocr_mobilev2_det_%s.onnx'%lang)
    globals()['rec_net'] = rt.InferenceSession(root+'/ppocr_mobilev2_rec_%s.onnx'%lang)
    globals()['cls_net'] = rt.InferenceSession(root+'/ppocr_mobilev2_cls_all.onnx')

# get mask
@tile(glob=32)
def get_mask(img):
    img = img[:,:,:3].astype('float32')/255
    offset = [0.485, 0.456, 0.406]
    offset = np.array(offset, dtype=np.float32)
    img = img - offset[None,None,:]
    img /= np.array([0.229, 0.224, 0.225])
    img = img.transpose(2,0,1)[None,:]
    return det_net.run(None, {'x':img})[0][0,0]

# find boxes from mask, and filter the bad boxes
def db_box(hot, thr=0.3, boxthr=0.7, sizethr=5, ratio=2):
    lab, n = ndimg.label(hot > thr)
    idx = np.arange(n) + 1
    level = ndimg.mean(hot, lab, idx)
    objs = ndimg.find_objects(lab, n)
    boxes = []
    for i, l, sli in zip(idx, level, objs):
        if l < boxthr: continue
        rcs = np.array(np.where(lab[sli]==i)).T
        if rcs.shape[0] < sizethr**2: continue
        o = rcs.mean(axis=0); rcs = rcs - o
        vs, ds = np.linalg.eig(np.cov(rcs.T))
        if max(vs)/min(vs)<2:
            vs = rcs.var(axis=0)
            ds = -np.eye(2)
        elif vs[0]>vs[1]:
            vs, ds = vs[[1,0]], ds[:,[1,0]]
        if ds[0,1]<0: ds[:,1] *= -1
        if np.cross(ds[:,0], ds[:,1])>0:
            ds[:,0] *= -1
        mar = vs.min() ** 0.5 * ratio * 2
        rcs = np.linalg.inv(ds) @ rcs.T
        minr, minc = rcs.min(axis=1) - mar
        maxr, maxc = rcs.max(axis=1) + mar
        if rcs.ptp(axis=1).min()<sizethr: continue
        rs = [minr,minc,minr,maxc,
              maxr,maxc,maxr,minc]
        rec = ds @ np.array(rs).reshape(-1,2,1)
        o += sli[0].start, sli[1].start
        rec = rec.reshape(4,2) + o
        first = np.argmin(rec.sum(axis=1))
        if vs[1]/vs[0]>2 and first%2==0: first+=1
        boxes.append(rec[(np.arange(5)+first)%4])
    return np.array(boxes)

# extract text image from given box
def extract(img, box, height=32):
    h = ((box[1]-box[0])**2).sum()**0.5
    w = ((box[2]-box[1])**2).sum()**0.5
    h, w = height, int(height * w / h)
    rr = box[[0,3,1,2],0].reshape(2,2)
    cc = box[[0,3,1,2],1].reshape(2,2)
    rcs = np.mgrid[0:1:h*1j, 0:1:w*1j]
    r2 = mapcoord(rr, *rcs, backend=np)
    c2 = mapcoord(cc, *rcs, backend=np)
    return mapcoord(img, r2, c2, backend=np)

# batch extract by boxes
def extracts(img, boxes, height=32, mwidth=0):
    rst = []
    for box in boxes:
        temp = extract(img, box, height)
        temp = temp.astype(np.float32)
        temp /= 128; temp -= 1
        rst.append(temp)
    ws = np.array([i.shape[1] for i in rst])
    maxw = max([i.shape[1] for i in rst])
    for i in range(len(rst)):
        mar = maxw - rst[i].shape[1] + 10
        rst[i] = np.pad(rst[i], [(0,0),(0,mar),(0,0)])
        if mwidth>0: rst[i] = rst[i][:,:mwidth]
    return np.array(rst).transpose(0,3,1,2), ws

# direction fixed
def fixdir(img, boxes):
    x, ws = extracts(img, boxes, 48, 256)
    y = cls_net.run(None, {'x':x})[0]
    dirs = np.argmax(y, axis=1)
    prob = np.max(y, axis=1)
    for b,d,p in zip(boxes, dirs, prob):
        if d and p>0.9: b[:] = b[[2,3,0,1,2]]
    return dirs, np.max(y, axis=1)

# decode
def ctc_decode(x, blank=10):
    x, p = x.argmax(axis=1), x.max(axis=1)
    if x.max()==0: return 'nothing', 0
    sep = (np.diff(x, prepend=[-1]) != 0)
    lut = np.where(sep)[0][np.cumsum(sep)-1]
    cal = np.arange(len(lut)) - (lut-1)
    msk = np.hstack((sep[1:], x[-1:]>0))
    msk = (msk>0) & ((x>0) | (cal>blank))
    cont = ''.join(lab_dic[x[msk]-1])
    return cont, p[msk].mean()
    
# recognize and decode every tensor
def recognize(img, boxes):
    x, ws = extracts(img, boxes, 32)
    cls = ws // 256
    rsts = ['nothing'] * len(boxes)
    for level in range(cls.max()+1):
        idx = np.where(cls==level)[0]
        if len(idx)==0: continue
        subx = x[idx,:,:,:(level+1) * 256]
        y = rec_net.run(None, {'x':subx})[0]
        for i in range(len(y)):
            rsts[idx[i]] = ctc_decode(y[i])
    return rsts

def ocr(img, autodir=False, thr=0.3, boxthr=0.7,
        sizethr=5, ratio=1.5, prothr=0.6, sample=1):
    hot = get_mask(img, sample=sample)
    boxes = db_box(hot, thr, boxthr, sizethr, ratio)
    if len(boxes)==0: return []
    if autodir: fixdir(img, boxes)
    box_cont = zip(boxes, recognize(img, boxes))
    rst = [(b.tolist(), *sp) for b,sp in box_cont]
    return [i for i in rst if i[2]>prothr]
    
def show(img, conts):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    for b, s, p in conts:
        b = np.array(b)
        plt.plot(*b.T[::-1], 'blue')
        plt.text(*b[0,::-1]-5, s, color='red')
    plt.show()

def test():
    import matplotlib.pyplot as plt
    from imageio import imread
    
    img = imread(root + '/card.jpg')[:,:,:3]
    from skimage.data import page
    
    conts = ocr(img, autodir=True)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.imshow(img)
    for b,s,p in conts:
        b = np.array(b)
        plt.plot(*b.T[::-1], 'blue')
        plt.text(*b[0,::-1]-5, s, color='red')
    plt.show()

if __name__ == '__main__':
    load()
    test()

    
    

    
