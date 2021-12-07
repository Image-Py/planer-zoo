import numpy as np
import planer

root = '/'.join(__file__.split('\\')[:-1])+'/models'
def load(name='u2netp_general'): 
    globals()['net'] = planer.read_net(root+'/%s.onnx'%name)
    
def global_norm(img):
    img.shape = img.shape[:2]+(-1,)
    img = img.astype(np.float32)
    img -= img.min(); img /= img.max()
    offset = [0.485, 0.456, 0.406]
    offset = np.array(offset, dtype=np.float32)
    img = img - offset[None,None,:]
    img /= np.array([0.229, 0.224, 0.225])
    return img

@planer.tile(glob=32)
def matting(img):
    img = img.transpose(2,0,1)[None,:,:,:]
    d1,d2,d3,d4,d5,d6,d7 = net(img)
    rst = d1[0,0,:,:]
    rst -= rst.min()
    rst /= rst.max()
    return rst

def test():
    from imageio import imread
    import matplotlib.pyplot as plt
    img = imread(root+'/lion.jpg')
    normimg = global_norm(img)
    rst = matting(normimg, sample=1, window=512)
    plt.subplot(121).imshow(img)
    plt.subplot(122).imshow(1-rst, cmap='gray')
    plt.show()
    
if __name__ == '__main__':
    load()
    test()
