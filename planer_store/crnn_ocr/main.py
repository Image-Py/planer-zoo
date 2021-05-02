import planer
np = planer.pal

source = ['https://gitee.com/imagepy/planer-store/attach_files/689491/download/crnn-ocr.pla',
          'https://gitee.com/imagepy/planer-store/attach_files/689699/download/planer.jpg']

root = '/'.join(__file__.split('\\')[:-1])

dic = ''' !@#$%&()+-:,.'"/\\?0123456789abcdefghijklmnopqrstuvwxyz'''

def load():
    globals()['net'] = planer.read_net(root+'/crnn-ocr')
    
def greedy_search(raw, blank=0):
    max_id = raw.argmax(2).ravel()
    msk = max_id[1:] != max_id[:-1]
    max_id = max_id[1:][msk]
    return max_id[max_id!=blank]

def recognize(img):
    img = planer.asarray(img)
    x = img.astype('float32')
    if x.ndim==3: x=x.mean(-1)
    w = 48*x.shape[1]//x.shape[0]
    x = planer.resize(x, (48, w))
    x = (x - 0.5)/(90/255)
    y = net(x[None, None, :, :])
    pred = greedy_search(y)
    pred = [dic[i] for i in pred.tolist()]
    return ''.join(pred)

def show():
    import matplotlib.pyplot as plt
    from PIL import Image
    img = Image.open(root+'/planer.jpg')
    img = np.asnumpy(img)
    text = recognize(img)
    plt.imshow(img)
    plt.title(text)
    plt.show()
    
if __name__ == '__main__':
    load()
    show()
