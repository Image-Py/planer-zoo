import os, os.path as osp
import importlib, planer
from tqdm import tqdm
import urllib.request

root = osp.abspath(osp.dirname(__file__)).replace('\\','/')

def progress(i, n, bar=[None]):
    if bar[0] is None:
        bar[0] = tqdm()
    bar[0].total = n
    bar[0].update(i-bar[0].n)
    if n==i: bar[0] = None
   
def download(url, path, info=print, progress=progress):
    info('download from %s'%url)
    f, rst = urllib.request.urlretrieve(url, path,
        lambda a,b,c: progress(int(100.0 * a * b/c), 100))

def downloads(model, names='required', force=False, info=print, progress=progress):
    lst = model.source()
    if names=='all': lst = [i for i in lst]
    elif names=='required': lst = [i for i in lst if i[1]]
    else:
        if isinstance(names, str): names = [names]
        lst = [i for i in lst if i[0] in names]
    if not force: lst = [i for i in lst if not i[2]]

    name = '/'.join(model.__package__.split('.')[1:])
    path = root + '/' + name + '/models/'
    if not osp.exists(path): os.mkdir(path)
    for name, a, b, url in lst: download(url, path+name, info, progress)

def list_source(model):
    print('%-20s%-10s%-10s'%('file name','installed', 'required'))
    print('-'*40)
    for i in model.source():
        print('%-20s%-10s%-10s'%(tuple(i[:3])))

def source(model, lst):
    name = '/'.join(model.__package__.split('.')[1:])
    for i in lst:
        i[2] = osp.exists(root + '/' + name + '/models/'+i[0])
    return lst
    
# name, need, has, url
def get_source(path):
    with open(path) as f: cont = f.read().split('\n')
    status, files = False, []
    for i in range(len(cont)):
        if '|File|' in cont[i].replace(' ',''): break
    for i in range(i, len(cont)):
        if not '|' in cont[i]: break
        if not '](' in cont[i]: continue
        nameurl = cont[i].split('|')[1]
        req = cont[i].split('|')[2].strip()!=''
        name, url = nameurl.split('](')
        name = name.split('[')[1]
        url = url.split(')')[0]
        files.append([name, req, False, url])
    root = osp.split(path)[0]
    return files

def load_model(name, auto=True):
    model = importlib.import_module('.'+name, 'planer_zoo')
    if '__init__.py' in model.__file__: name += '.readme'
    md = root + '/' +name.replace('.', '/') + '.md'
    model.source = lambda m=model: source(m, get_source(md))
    model.list_source = lambda x=model: list_source(x)
    name = '/'.join(model.__package__.split('.')[1:])
    model.download = lambda name='required', force=False, m=model: downloads(m, name, force)
    if auto: model.download()
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
    a = list_source('./yolo/v5/yolo5.md')
