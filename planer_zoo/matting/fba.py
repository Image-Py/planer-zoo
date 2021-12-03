import planer
from planer import tile
# import onnxruntime as rt

from scipy.ndimage import distance_transform_edt as edt

root = '/'.join(__file__.split('\\')[:-1])+'/models'
def load(): 
    globals()['net'] = planer.read_net(root+'/fba.onnx')

def prework(img, mark):
    import numpy as np
    img = img.astype(np.float32)/255
    mark = mark.astype(np.float32)/255
    
    trimap = np.array([mark==0, mark==1])
    trimap = trimap.astype(np.float32)

    offset = np.array([0.485, 0.456, 0.406])
    nimg = img - offset.astype('float32')[None,None,:]
    nimg /= np.array([0.229, 0.224, 0.225])

    L, w = 320, [0.02, 0.08, 0.16]
    k = 2 * (np.array(w)*L)[:,None,None]**2
    dis0 = edt(trimap[0]!=1); dis0 **= 2
    dis1 = edt(trimap[1]!=1); dis1 **= 2
    y = np.vstack((dis0[None,:]/k, dis1[None,:]/k))
    ntrimap = np.exp(-y, out=y).astype(np.float32)

    return np.concatenate([
        img, trimap.transpose(1,2,0),
        nimg, ntrimap.transpose(1,2,0)], axis=-1)
    
@tile(glob=48)
def matting(img_mark):
    img_mark = img_mark.transpose(2,0,1)[None,:]
    img = img_mark[:,0:3]
    trimap = img_mark[:,3:5]
    nimg = img_mark[:,5:8]
    ntrimap = img_mark[:,8:14]
    para = {'img':img, '1':trimap, '2':nimg, '3':ntrimap}
    y = net.run(None, para)[0]
    return y[0].transpose(1,2,0)

def test():
    from imageio import imread
    import matplotlib.pyplot as plt

    img = imread(root+'/image.png')
    mark = imread(root+'/trimap.png')
    img_mark = prework(img, mark)
    y = matting(img_mark)

    plt.subplot(221).imshow(img)
    plt.subplot(222).imshow(y[:,:,0])
    plt.subplot(223).imshow(y[:,:,1:4])
    plt.subplot(224).imshow(y[:,:,4:7])
    plt.show()
    
if __name__ == '__main__':
    load()
    test()
