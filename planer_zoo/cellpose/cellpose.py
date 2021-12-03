import planer
import numpy as np
import scipy.ndimage as ndimg
from time import time

root = '/'.join(__file__.split('\\')[:-1])+'/models'
def load(name='cyto_v1'):
    globals()['net'] = planer.read_net(root+'/%s.onnx'%name)

@planer.tile(glob=64)
def count_flow(img):
    img = img[None, :, :]/255
    img = np.concatenate((img, img*0))
    y, style = net(img[None,:,:,:])
    y[0,2] = 1/(1+np.e**-y[0,2])
    return y[0].transpose(1,2,0)

def estimate_volumes(arr, sigma=3):
    msk = arr > 50; 
    idx = np.arange(len(arr), dtype=np.uint32)
    idx, arr = idx[msk], arr[msk]
    for k in np.linspace(5, sigma, 5):
       std = arr.std()
       dif = np.abs(arr - arr.mean())
       msk = dif < std * k
       idx, arr = idx[msk], arr[msk]
    return arr.mean(), arr.std()

def flow2msk(flowp, level=0.5, grad=0.5, area=None, volume=None):
    shp, dim = flowp.shape[:-1], flowp.ndim - 1
    l = np.linalg.norm(flowp[:,:,:2], axis=-1)
    flow = flowp[:,:,:2]/l.reshape(shp+(1,))
    flow[(flowp[:,:,2]<level)|(l<grad)] = 0
    ss = ((slice(None),) * (dim) + ([0,-1],)) * 2
    for i in range(dim):flow[ss[dim-i:-i-2]+(i,)]=0
    sn = np.sign(flow); sn *= 0.5; flow += sn;
    dn = flow.astype(np.int32).reshape(-1, dim)
    strides = np.cumprod(np.array((1,)+shp[::-1]))
    dn = (strides[-2::-1] * dn).sum(axis=-1)
    rst = np.arange(flow.size//dim); rst += dn
    for i in range(10): rst = rst[rst]
    hist = np.bincount(rst, None, len(rst))
    hist = hist.astype(np.uint32).reshape(shp)
    lab, n = ndimg.label(hist, np.ones((3,)*dim))
    volumes = ndimg.sum(hist, lab, np.arange(n+1))
    areas = np.bincount(lab.ravel())
    mean, std = estimate_volumes(volumes, 2)
    if not volume: volume = max(mean-std*3, 50)
    if not area: area = volumes // 3
    msk = (areas<area) & (volumes>volume)
    lut = np.zeros(n+1, np.uint32)
    lut[msk] = np.arange(1, msk.sum()+1)
    return lut[lab].ravel()[rst].reshape(shp)

def flow2hsv(flow):
    v = np.linalg.norm(flow, axis=-1)
    h = np.arccos(flow[:,:,0]/v)
    h *= np.sign(flow[:,:,1])/(np.pi*2)
    h += 0.5; v /= v.max()
    s = np.ones_like(v)             
    a = np.floor(h * 6)
    b = h * 6; b -= a
    p = np.zeros_like(v)
    t = v * b; q = v - t
    buf = np.stack((v,t,p,q), -1).ravel()
    buf *= 255; buf = buf.astype(np.uint8)
    idx = np.array([[0,1,3],[3,0,2],
        [2,0,1], [2,3,0],[1,2,0],[0,2,3]])
    idx = idx[a.ravel().astype(np.uint8) % 6]
    idx += np.arange(v.size)[:,None] * 4
    return buf[idx].reshape(v.shape+(3,))

def draw_edge(img, lab, color=(255,0,0)):
    msk = np.zeros(lab.shape, dtype=np.bool)
    mskr = lab[1:] != lab[:-1]
    mskc = lab[:,1:] != lab[:,:-1]
    msk[1:] |= mskr; msk[:-1] |= mskr
    msk[:,1:] |= mskc; msk[:,:-1] |= mskc
    lut = np.array([[0,0,0],color], dtype=np.uint8)
    rgb = lut[msk.view(np.uint8)]
    img = img.reshape((img.shape+(1,))[:3])
    return np.maximum(img, rgb, out=rgb)

def rgb_mask(img, lab):
    cmap = np.array([(0,0,0), (255,0,0),
        (0,255,0), (0,0,255), (255,255,0),
        (255,0,255), (0,255,255)], np.uint8)
    msk = lab > 0; lab %= 6; lab += 1
    rgb = cmap[lab * msk]
    img = img.reshape((img.shape+(1,))[:3])
    return np.maximum(img, rgb, out=rgb)

def test():
    import matplotlib.pyplot as plt
    from imageio import imread
    img = imread(root + '/cell.png')[:,:,0]

    flow = count_flow(img, sample=1, window=512, margin=0.1)
    lab = flow2msk(flow, level=0.2)
    edge = draw_edge(img, lab)
    rgb = rgb_mask(img, lab)

    plt.subplot(221).imshow(img)
    plt.subplot(222).imshow(flow2hsv(flow))
    plt.subplot(223).imshow(edge)
    plt.subplot(224).imshow(rgb)
    plt.show()
    
if __name__ == '__main__':
    load()
    test()
    
