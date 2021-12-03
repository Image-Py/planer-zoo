import planer as rt
import numpy as np

root = '/'.join(__file__.split('\\')[:-1])+'/models'
def load(): 
    globals()['net'] = rt.InferenceSession(root+'/real_esrgan_x4.onnx')

@rt.tile(window=512)
def enhance(img):
    img = img.astype(np.float32)/255
    img = img.transpose(2,0,1)[None,:]
    y = net.run(None, {'input.1':img})[0]
    y = y[0].transpose(1,2,0)
    y = np.clip(y, 0, 1, out = y)
    return (y * 255).astype(np.uint8)
    
def test():
    from imageio import imread
    import matplotlib.pyplot as plt

    img = imread(root + '/mushroom.jpg')[::3,::3]
    y = enhance(img)

    plt.subplot(121).imshow(img)
    plt.subplot(122).imshow(y)
    plt.show()

if __name__ == '__main__':
    test()
