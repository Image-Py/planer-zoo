import planer, json
import numpy as np

root = '/'.join(__file__.split('\\')[:-1])+'/models'

def load(name='resnet'):
    globals()['net'] = planer.read_net(root+'/%s.onnx'%name)
    with open(root+'/imagenet_labs.json') as f:
        globals()['classes'] = json.load(f)

def recognize(img):
    img = planer.asarray(img)
    x = planer.resize(img, (224, 224))
    x = x[None, :,:,:].astype('float32')
    x = (x/255).transpose(0, 3, 1, 2)
    y = np.argmax(net(x), axis=-1)
    return classes[str(y[0])]

def test():
    import matplotlib.pyplot as plt
    from imageio import imread
    img = imread(root+'/bus.jpg')
    img = np.asnumpy(img)
    rst = recognize(img)
    plt.imshow(img)
    plt.title(rst)
    plt.show()
    
if __name__ == '__main__':
    load('mobilenet')
    test()
