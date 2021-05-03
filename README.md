# planer-store
[Planer](https://github.com/Image-Py/planer) is a light-weight CNN framework implemented in pure Numpy-like interface. It can run only with Numpy. Or change different backends. (Cupy accelerated with CUDA, ClPy accelerated with OpenCL).

planer-store is a toolbox based on planer. Supportting many models, we can use them easily:
```python
import planer_store as plas
model = plas.load_model('xxx')

model.xxx ...
```

## crnn-ocr
```python
import planer_store as plas
model = plas.load_model('crnn_ocr')

img = imread('planer.jpg')
text = model.recognize(img)

# show img and result
```
![crnn-ocr](https://user-images.githubusercontent.com/24822467/116868860-25630f00-ac42-11eb-92ce-a645eb7d5a8a.png)

## face-68-key
```python
import planer_store as plas
model = plas.load_model('face_68_key')

face = imread('face.jpg')
rc = model.get_face_key(face)
thin = model.face_adjust(face, fac=-10)
fat = model.face_adjust(face, fac=10)

# show face, thin, fat
```
![face-69key](https://user-images.githubusercontent.com/24822467/116868863-25fba580-ac42-11eb-82fb-4896162b70af.png)

## hed-edge
```python
import planer_store as plas
model = plas.load_model('hed_edge')

face = imread('edge.jpg')
edge = model.hed_edge(img)

# show face, edge
```
![hed-edge](https://user-images.githubusercontent.com/24822467/116868866-26943c00-ac42-11eb-9b6e-b06776a67659.png)

## object-recognize
```python
import planer_store as plas
model = plas.load_model('resnet18')

img = imread('bus.jpg')
obj = model.recognize(img)

# show img and result
```
![object-recognize](https://user-images.githubusercontent.com/24822467/116868854-23994b80-ac42-11eb-81ce-31b945a09ccc.png)

## high-resolution
```python
import planer_store as plas
model = plas.load_model('ESRGAN')

img = imread('girl.jpg')
high = model.super_resolution(img)

# show img and high resolution result
```
![high-resolution](https://user-images.githubusercontent.com/24822467/116868867-26943c00-ac42-11eb-96b8-844f44aa63ac.png)

## cellpose
```python
import planer_store as plas
model = plas.load_model('cellpose')

img = 255 - imread('cell.png')[:,:,0]
flow = count_flow(img)
lab = flow2msk(flow, level=0.2)
edge = draw_edge(img, lab)
rgb = rgb_mask(img, lab)

# show img, flow, edge, rgbmsk
```
![cellpose](https://user-images.githubusercontent.com/24822467/116868857-24ca7880-ac42-11eb-8799-37f65a1e2421.png)

## Contribution
welcom to contribute new models.
1. use torch to train, and export as onnx
2. planer.onnx2planer('xxx.onnx')