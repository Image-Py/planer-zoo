import os, os.path as osp
import importlib, planer
from tqdm import tqdm
import urllib.request

root = osp.abspath(osp.dirname(__file__))

def progress(i, n, bar=[None]):
    if bar[0] is None:
        bar[0] = tqdm()
    bar[0].total = n
    bar[0].update(i-bar[0].n)
    if n==i: bar[0] = None
   
def download(url, path, info=print, progress=progress):
    info('download from %s'%url)
    urllib.request.urlretrieve(url, path,
        lambda a,b,c: progress(int(100.0 * a * b/c), 100))

def downloads(src, path, info=print, progress=progress):
    for i in src: download(i, path+'/'+osp.split(i)[-1], info, progress)
      
def load_model(name, auto=True):
    model = importlib.import_module('.'+name+'.main', 'planer_store')
    model.download = lambda : downloads(model.source, root+'/'+name)
    has = lambda i: osp.exists(root+'/'+name+'/'+osp.split(i)[-1])
    if auto: downloads([i for i in model.source if not has(i)], root+'/'+name)
    model.load()
    return model

def core(obj): planer.core(obj)

def test():
    load_model('crnn_ocr').show()
    load_model('face_68_key').show()
    load_model('hed_edge').show()
    load_model('obj_recognize').show()
    load_model('super_resolution').show()
    load_model('cellpose').show()

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    test()
