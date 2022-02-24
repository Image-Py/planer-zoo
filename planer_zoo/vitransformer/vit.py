import planer as ort
import planer
import json
import numpy as np
from time import time


root = '/'.join(__file__.split('\\')[:-1])+'/models'

def load():
    globals()['net'] = planer.read_net(root+'/vit.onnx')
    with open(root+'/imagenet_labs.json') as f:
        globals()['classes'] = json.load(f)

def recognize(img):
    img = planer.asarray(img)
    x = planer.resize(img, (384,384))
    x = x.transpose(2,0,1)[None,:]
    x = x.astype('float32') / 255
    x -= 0.5; x *= 2
    y = np.argmax(net(x), axis=-1)
    return classes[str(y[0])]

def test():
    import matplotlib.pyplot as plt
    from imageio import imread
    img = imread(root+'/pandas.jpg')
    img = np.asnumpy(img)
    rst = recognize(img)
    plt.imshow(img)
    plt.title(rst)
    plt.show()

if __name__ == '__main__':
    load()
    test()

