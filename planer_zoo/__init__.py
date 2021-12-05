import importlib, planer

def load_model(name, auto=True):
    model = importlib.import_module('.'+name, 'planer_zoo')
    return planer.Model(model, auto)

if __name__ == '__main__':
    a = list_source('./yolo/v5/yolo5.md')
