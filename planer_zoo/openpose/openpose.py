import planer
import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

root = '/'.join(__file__.split('\\')[:-1])+'/models'

def load(lang='ch'):
    globals()['net'] = planer.read_net(root+'/openpose')
    
limbidx = np.array([1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 1,
    8, 8, 9, 9, 10, 1, 11, 11, 12, 12, 13, 1, 0, 0, 14, 14,
    16, 0, 15, 15, 17]).reshape(-1,2) # '''2, 16, 5, 17'''

mapidx = np.array([12, 13, 20, 21, 14, 15, 16, 17, 22,
    23, 24, 25, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    28, 29, 30, 31, 34, 35, 32, 33, 36, 37, 18, 19,
    26, 27]).reshape(-1,2)

# count the body flow hotmap
def get_flow(image, scale=(400,600)):
    shp = image.shape[:2]
    if isinstance(scale, float):
        scale = shp[0]*scale, shp[1]*scale
    image = planer.np.asarray(image)
    scale = int(scale[0]/8+0.5)*8, int(scale[1]/8+0.5)*8
    k = shp[0]/scale[0], shp[1]/scale[1]
    img = planer.resize(image, scale)
    img = img.transpose(2,0,1)[None,:]/255
    return net.run(None, {'inputc': img}) +(k,)
    
# peaks: ns, cn, r, c, score, id, personid
def get_peaks(maxmsks, heatmaps, thr=0.1, lim=30):
    np = planer.np # use the same backend
    maxmsks *= heatmaps > thr
    size = maxmsks.shape[0]
    nb, cn, row, col = np.where(maxmsks[:,:-1])
    if len(cn)==0: return None, None
    score = (heatmaps[nb, cn, row, col] * 100)
    bins = np.bincount(nb, minlength=size)
    ids = np.arange(len(cn), dtype='int32')
    ids -= np.hstack(([0], np.cumsum(bins)))[nb]
    crsitp = (nb, cn, col, row, score, ids, cn)
    peaks = np.array(crsitp).astype('int32').T
    bins = np.bincount(nb*19+cn, minlength=19*size)
    lim = min(int(bins.max()), lim)
    msk = np.arange(lim) >= bins[:,None]
    start = np.hstack(([0], np.cumsum(bins)[:-1]))
    idx = np.arange(lim) + start[:,None]
    idx[msk] = -1; idx = idx.ravel()
    speaks = peaks[idx] * (idx[:,None]>-1)
    return peaks, speaks.reshape(size, 19, lim, -1)

# conns: n, cn, p1id, p2id, g1id, g2id, linescore, pathscore
def get_conns(flows, peaks, npts=10, thr=0.05):
    np = planer.np # use the same backend
    n, c, h, w = flows.shape
    # layer, ab, point, xy, lins1
    lins1 = np.linspace(0, 1, npts)
    lins2 = np.linspace(1, 0, npts)
    # n, layer, cn, h, w
    flows = flows[:,mapidx]
    # n, layer, ab, point, xy, lins1
    relmat = peaks[:,limbidx, :, 2:4, None]
    ca = relmat[:,:,0,:,None] * lins1
    cb = relmat[:,:,1,None,:] * lins2
    # n, cn, pa, pb, xy, lins
    cab = ca + cb # connection a to b
    dv = cab[...,0] - cab[...,1]
    cab = np.around(cab).astype('int32')
    ndv = np.linalg.norm(dv, axis=-1)
    np.maximum(ndv, 0.001, out=ndv)
    dv /= ndv[...,None]
    nidx = np.zeros_like(cab)
    nax, nbx = nidx[...,1,:], nidx[...,0,:]
    nax[:] = np.arange(n).reshape(-1,1,1,1,1)
    nbx[:] = np.arange(17).reshape(1,-1,1,1,1)
    cax, cbx = cab[...,1,:], cab[...,0,:]
    vs = flows[nax, nbx, :, cax, cbx]
    vs = (vs * dv[...,None,:]).sum(-1)
    candmsk = (vs>thr).sum(axis=-1)
    adj = 0.5 * flows.shape[3] / ndv - 1
    vs = vs.mean(-1) + np.minimum(adj, 0)
    candmsk = candmsk > 0.8 * npts
    n, cn, r, c = np.where(candmsk & (vs > 0))
    na, nb = np.asarray(limbidx)[cn].T
    candv = vs[n, cn,r,c] * 200
    nax, nbx = peaks[n,na,r,4:6], peaks[n,nb,c,4:6]
    pathv = candv + nax[:,0] + nbx[:,0]
    rs = n, cn, nax[:,1], nbx[:,1], r, c, candv, pathv
    return np.array(rs, dtype='int32').T

# split stack in limb
def split(arrs, c, l):
    bins = np.bincount(arrs[:,c], minlength=l)
    seps = np.cumsum(bins).tolist()
    return np.split(arrs, seps[:-1])
        
# assign the peak pair
def assignment(conns):
    for part in range(19):
        conn = conns[part]
        if len(conn)==0: continue
        shape = conn[:,[4,5]].max(axis=0)+1
        buf = np.zeros(shape, 'int32')
        buf[conn[:,4], conn[:,5]] = conn[:,7]
        buf[linear_sum_assignment(-buf)] = 0
        msk = buf[conn[:,4], conn[:,5]]==0
        conns[part] = conn[msk]
    return conns

# combine peaks by connection map
def combine(speaks, conns, scale):
    # speaks = numpy.vstack(peaks)
    sconns = np.vstack(conns)
    buf = np.zeros((speaks.shape[0],)*2, 'uint8')
    buf[sconns[:,2], sconns[:,3]] = sconns[:,6]
    g = csr_matrix(buf)
    n, lab = connected_components(g, False)
    conslab = lab[sconns[:,2]]
    speaks[:,-1] = lab
    counts = np.bincount(lab)
    color = np.where(counts>8)[0]
    bodys = []
    for c in color:
        relab = np.cumsum(lab==c)-1
        pts = speaks[lab==c]
        pts[:,2:4] = pts[:,2:4] * scale[::-1]
        con = sconns[conslab==c]
        con[:,4:6] = relab[con[:,2:4]]
        bodys.append((pts, con))
    return bodys

# build bodys from heatmap and flowmap
def make_body(heats, flows, maxmsks, thr_peak=0.1, thr_flow=0.05, scale=1, lim=50):
    peaks, speaks = get_peaks(maxmsks, heats, lim=lim)
    if peaks is None: return []
    conns = get_conns(flows, speaks)
    if not isinstance(conns, np.ndarray):
        peaks, conns = peaks.get(), conns.get()
    
    npeaks = split(peaks, 0, len(heats))
    nconns = split(conns, 0, len(heats))
    bodys = []
    for cpeak, cconn in zip(npeaks, nconns):
        ncconn = split(cconn, 1, 19)
        body = combine(cpeak, assignment(ncconn), scale)
        bodys.append(body)
    return bodys

# draw image and body
def draw(img, bodys):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    colors = ['blue', 'yellow', 'green', 'orange', 'cyan', 'gray']
    for i in range(len(bodys)):
        pts, conn = bodys[i]
        plt.plot(*pts[:,2:4].T, 'r.')
        for i1, i2 in conn[:,4:6]:
            plt.plot(*pts[[i1,i2],2:4].T, colors[i%6])
    plt.show()

def test():
    from imageio import imread
    oriimg = imread(root+'/bodys.jpg')
    flow, heat, maxheat, k = get_flow(oriimg, (400,600))
    bodys = make_body(heat, flow, heat==maxheat, scale=k)
    draw(oriimg, bodys[0])
    
if __name__ == '__main__':
    load()
    test()

    
