import planer, json
np = planer.pal

source = ['https://gitee.com/imagepy/planer-store/attach_files/690786/download/resnet18.pla',
          'https://gitee.com/imagepy/planer-store/attach_files/690789/download/imagenet_labs.json',
          'https://gitee.com/imagepy/planer-store/attach_files/690790/download/bus.jpg']

root = '/'.join(__file__.split('\\')[:-1])

def load():
    globals()['net'] = planer.read_net(root+'/resnet18')
    with open(root+'/imagenet_labs.json') as f:
        globals()['classes'] = json.load(f)

def recognize(img):
    img = planer.asarray(img)
    x = planer.resize(img, (224, 224))
    x = x[None, :,:,:].astype('float32')
    x = (x/255).transpose(0, 3, 1, 2)
    y = np.argmax(net(x), axis=-1)
    return classes[str(y)]

def show():
    import matplotlib.pyplot as plt
    from PIL import Image
    img = Image.open(root+'/bus.jpg')
    img = np.asnumpy(img)
    rst = recognize(img)
    plt.imshow(img.astype('uint8'))
    plt.title(rst)
    plt.show()
    
if __name__ == '__main__':
    load()
    show()
