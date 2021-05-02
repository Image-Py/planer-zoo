import planer, json
np = planer.pal

source = ['https://gitee.com/imagepy/planer-store/attach_files/690791/download/ESRGAN.pla',
          'https://gitee.com/imagepy/planer-store/attach_files/690797/download/girl.png']

root = '/'.join(__file__.split('\\')[:-1])

def load():
    globals()['net'] = planer.read_net(root+'/ESRGAN')


def super_resolution(img):
    img = planer.asarray(img)
    x = (img/255.0).transpose(2, 0, 1)
    x = x[None, :, :, :].astype('float32')
    y = net(x).transpose(1, 2, 0)
    return planer.asnumpy(y)
    
def show():
    import matplotlib.pyplot as plt
    from PIL import Image
    img = Image.open(root+'/girl.png')
    img = np.asnumpy(img)
    rst = super_resolution(img)
    plt.subplot(121).imshow(img.astype('uint8'))
    plt.subplot(122).imshow(rst.clip(0,1))
    plt.show()
    
if __name__ == '__main__':
    load()
    show()
