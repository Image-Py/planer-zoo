import planer
np = planer.pal

source = ['https://gitee.com/imagepy/planer-store/attach_files/690801/download/hed.pla',
          'https://gitee.com/imagepy/planer-store/attach_files/690800/download/plane.jpg']

root = '/'.join(__file__.split('\\')[:-1])

def load():
    globals()['net'] = planer.read_net(root+'/hed')

def hed_edge(img, size=64):
    img = planer.asarray(img, dtype='float32')
    x.reshape((-1, 3))[:] -= np.array([104, 117, 123])
    x = x.transpose(2, 0, 1)[None, :, :, :]
    return np.asnumpy(planer.resize(net(x)[0], (h,w)))

@planer.tile(glob=32)
def hed_edge(img):
    x = img.transpose(2,0,1)[None, :, :, :]
    d = np.array([104, 117, 123])[:,None,None]
    return net(x - d)[0]

def show():
    import matplotlib.pyplot as plt
    from PIL import Image
    img = Image.open(root+'/plane.jpg')
    img = planer.asnumpy(img)
    edge = hed_edge(img)
    plt.subplot(121).imshow(img)
    plt.subplot(122).imshow(edge)
    plt.show()

if __name__ == '__main__':
    load()
    show()
    
