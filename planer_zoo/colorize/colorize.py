import planer as ort
from planer import tile
import numpy as np

root = '/'.join(__file__.split('\\')[:-1])+'/models'

def load(name='resnet'):
    globals()['net'] = ort.InferenceSession(root+'/colorize')

xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
                         [0.212671, 0.715160, 0.072169],
                         [0.019334, 0.119193, 0.950227]], 'float32')

rgb_from_xyz = np.linalg.inv(xyz_from_rgb)

def rgb2xyz(arr):
    arr = arr.astype('float32')
    msk, arr1 = arr > 0.04045, arr / 12.92
    arr += 0.055; arr /= 1.055; arr **= 2.4
    arr *= msk; arr1 *= ~msk; arr += arr1
    return arr @ xyz_from_rgb.T

def xyz2lab(arr):
    xyz_ref_white = [0.95047, 1., 1.08883]
    arr /= xyz_ref_white; msk = arr > 0.008856
    cbrt = np.cbrt(arr); arr *= 7.787; arr += 16/116; 
    cbrt *= msk; arr *= ~msk; arr += cbrt
    Lab = np.zeros(arr.shape, 'float32')
    l, a, b = Lab.transpose(2,0,1)
    l[:] = arr[:,:,1]; l *= 116; l -= 16
    a[:] = arr[:,:,0]; a -= arr[:,:,1]; a *= 500
    b[:] = arr[:,:,1]; b-= arr[:,:,2]; b *= 200
    return Lab
    
def rgb2lab(img): return xyz2lab(rgb2xyz(img))

def lab2xyz(lab):
    L, a, b = lab.transpose(2,0,1)
    xyz = np.zeros(lab.shape, 'float32')
    x, y, z = xyz.transpose(2,0,1)
    y[:] = L; y += 16; y /= 116
    x[:] = a; x /= 500; x += y
    z[:] = b; z /= -200; z += y
    np.maximum(z, 0, out=z)
    msk = xyz > 0.2068966
    cbp = xyz ** 3
    xyz -= 16/116; xyz /= 7.787
    xyz *= ~msk; cbp *= msk; xyz += cbp
    xyz *= [0.95047, 1., 1.08883]
    return xyz

def xyz2rgb(xyz):
    arr = xyz @ rgb_from_xyz.T
    msk = arr > 0.0031308
    arr1 = arr * msk; arr1 **= (1/2.4)
    arr1 *= 1.055; arr1 -= 0.055
    arr *= 12.92; arr1 *= msk;
    arr *= ~msk; arr += arr1
    return np.clip(arr, 0, 1, out=arr)

def lab2rgb(img): return xyz2rgb(lab2xyz(img))

@tile(glob=16)
def colorize(img):
    img /= 255
    lab = rgb2lab(img)[:,:,0]
    l = lab[None,None,:,:]
    ab = net.run(None, {'input':l})[0]
    lab = np.concatenate((l, ab), axis=1)[0]
    return lab2rgb(lab.transpose(1,2,0))

def test():
    from imageio import imread
    import matplotlib.pyplot as plt

    img = imread(root+'/bridge.jpg')
    colorimg = colorize(img, sample=0.5)
    plt.subplot(121).imshow(img)
    plt.subplot(122).imshow(colorimg)
    plt.show()

if __name__ == '__main__':
    load()
    test()
