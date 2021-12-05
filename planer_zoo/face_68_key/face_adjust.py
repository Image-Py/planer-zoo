import planer, os
import numpy as np

root = '/'.join(__file__.split('\\')[:-1])+'/models'

def load():
    globals()['net'] = planer.read_net(root+'/face68key.onnx')
    
# get pixels on the line
def line(r1, c1, r2, c2):
    d = max(abs(r1-r2), abs(c1-c2)) + 1
    rs = np.linspace(r1,r2,d).round()
    cs = np.linspace(c1,c2,d).round()
    return rs.astype(int), cs.astype(int)

# count flow from kay points
# weights: ear - face - jaw 's weights, 0 means stay there.
# fac: + means fat, - means thin, 0 would got a zeros flow
def count_flow(rc, weights=[0,1,2,2,2,2,1,1,0], fac=5):
    np = planer.np
    flow = np.zeros((224,224,2), dtype=np.float32)
    l = rc[:17].astype(np.int)
    nose = rc[31:36].mean(axis=0)
    dv = nose - l
    dv.T[:] *= weights + weights[1:]
    for s,e,v1,v2 in zip(l[:-1], l[1:], dv[:-1], dv[1:]):
        flow[line(*s, *e)] = (v1+v2)/2
    flow = planer.gaussian_filter(flow, 20)
    nv = np.linalg.norm(flow, axis=-1)
    return flow * (fac/nv.max())

# apply the flow
def flow_map(img, flow):
    np = planer.np
    des = np.mgrid[0:224, 0:224].transpose(1,2,0) + flow
    des = planer.resize(des, img.shape[:2])
    des *= np.array(img.shape[:2]).reshape(1,1,2)/224
    return planer.mapcoord(img, *des.transpose(2,0,1))

# generate check board
def check_board(shp, d=20):
    checkboard = np.zeros(shp, dtype=np.uint8)
    msk = np.tile([False]*d+[True]*d, 1000)
    checkboard[msk[:shp[0]]] += 128
    checkboard[:,msk[:shp[1]]] += 128
    return ((checkboard>0)*255).astype(np.uint8)

def get_face_key(img, scale=True):
    img = planer.asarray(img, dtype='float32')
    x = planer.resize(img, (224,224))/255
    if img.ndim==2: x = x[:,:,None]
    x = x.transpose(2,0,1)[None,:,:,:]
    rc = net(x).reshape(-1,2)[:,::-1] * 50 + 100
    rc = planer.asnumpy(rc)
    if scale: rc *= np.array(img.shape[:2])/224
    return rc

def face_adjust(img, weights=[0,1,2,2,2,2,1,1,0], fac=5):
    img = planer.asarray(img)
    rc = get_face_key(img, False)
    flow = count_flow(rc, weights, fac)
    return planer.asnumpy(flow_map(img, flow))

def test():
    import matplotlib.pyplot as plt
    from imageio import imread
    face = imread(root+'/benshan.jpg')

    rc = get_face_key(face)
    plt.subplot(131).imshow(face)
    plt.plot(*rc.T[::-1], 'r.')
    plt.title('68 key points')
    
    thin = face_adjust(face, fac=-10)
    plt.subplot(133).imshow(thin)
    plt.title('thin with fac -10')
    
    fat = face_adjust(face, fac=10)
    plt.subplot(132).imshow(fat)
    plt.title('fat with fac 10')
    plt.show()
    
if __name__ == '__main__':
    load()
    test()
    
