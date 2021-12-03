import planer
from planer import tile
import numpy as np


root = '/'.join(__file__.split('\\')[:-1])+'/models'

def load():
    globals()['A'] = planer.read_net(root+'/A_enc.onnx')
    globals()['B'] = planer.read_net(root+'/B_dec.onnx')
    globals()['mapping'] = planer.read_net(root+'/mapping.onnx')

@tile(glob=128, window=256)
def enhance(img):
    x = img.astype('float32')
    x = x.transpose(2,0,1)[None,:]
    x -= 0.5; x *= 2
    feat = A.run(None, {'input':x})[0]
    feat_map = mapping.run(None, {'input':feat})[0]
    y = B.run(None, {'input':feat_map})[0]
    y += 1; y /= 2
    return y[0].transpose(1,2,0)

def test():
    from matplotlib import pyplot as plt
    from imageio import imread
    
    img = imread(root+'/old_photo.png')
    y = enhance(img, sample=1, window=1024)

    plt.subplot(121)
    plt.imshow(img)

    plt.subplot(122)
    plt.imshow(y)
    plt.show()
    
if __name__ == '__main__':
    load()
    test()
