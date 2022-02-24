import importlib, planer

def load_model(name, auto=True):
    model = importlib.import_module('.'+name, 'planer_zoo')
    return planer.Model(model, auto)

def test():
    planer.load('planer_zoo.openpose').test()
    planer.load('planer_zoo.colorize').test()
    planer.load('planer_zoo.solo').test()
    planer.load('planer_zoo.vitransformer').test()
    planer.load('planer_zoo.ocr.paddleocr').test()
    planer.load('planer_zoo.deeplab').test()
    planer.load('planer_zoo.face_68_key').test()
    planer.load('planer_zoo.cellpose').test()
    planer.load('planer_zoo.matting.human').test()
    planer.load('planer_zoo.matting.u2net').test()
    planer.load('planer_zoo.matting.fba').test()
    planer.load('planer_zoo.obj_recognize').test()
    planer.load('planer_zoo.old_photos_life').test()
    planer.load('planer_zoo.super_resolution').test()
    planer.load('planer_zoo.yolo.yolov4').test()
    planer.load('planer_zoo.yolo.yolov5').test()

if __name__ == '__main__':
    test()
