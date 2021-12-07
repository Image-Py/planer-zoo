import numpy as np
import planer

root = '/'.join(__file__.split('\\')[:-1])+'/models'
def load(): 
    globals()['net'] = planer.read_net(root+'/rvm_mobilenetv3.onnx')
    
def global_norm(img):
    img.shape = img.shape[:2]+(-1,)
    img = img.astype(np.float32)
    img -= img.min()
    img /= img.max()
    return img

@planer.tile(glob=32)
def matting(img):
    img = img.transpose(2,0,1)[None,:,:,:]
    fbr, alpha, a, b, c, d = net(img)
    return alpha[0,0]

def test():
    from imageio import imread
    import matplotlib.pyplot as plt
    img = imread(root+'/astronaut.jpg')
    normimg = global_norm(img)
    rst = matting(normimg, sample=1, window=512)
    plt.subplot(121).imshow(img)
    plt.subplot(122).imshow(1-rst, cmap='gray')
    plt.show()
    
if __name__ == '__main__':
    load()
    test()
