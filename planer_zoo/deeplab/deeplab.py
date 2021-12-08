import planer
import numpy as np

root = '/'.join(__file__.split('\\')[:-1])+'/models'

def load():
    with open(root+'/names.txt') as f: globals()['labs'] = f.read().split('\n')
    globals()['net'] = planer.read_net(root+'/deeplabv3_resnet101.onnx')
    
def global_norm(img):
    img.shape = img.shape[:2]+(-1,)
    img = img.astype(np.float32)
    img -= img.min(); img /= img.max()
    offset = [0.485, 0.456, 0.406]
    offset = np.array(offset, dtype=np.float32)
    img = img - offset[None,None,:]
    img /= np.array([0.229, 0.224, 0.225])
    return img

@planer.tile(glob=32, window=1e5)
def matting(img):
    img = img.transpose(2,0,1)[None,:,:,:]
    out, aux = net(img)
    return out[0].argmax(axis=0)

def test():
    from imageio import imread
    import matplotlib.pyplot as plt
    img = imread(root+'/bus.jpg')
    normimg = global_norm(img)
    rst = matting(normimg, sample=0.5)
    plt.subplot(121).imshow(img)
    plt.subplot(122).imshow(rst)
    plt.show()
    
if __name__ == '__main__':
    load()
    test()
