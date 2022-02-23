from setuptools import setup, find_packages
import os

descr = """Some useful models based on planer"""

def get_data_files():
    dic = {}
    for root, dirs, files in os.walk('planer_zoo', True):
        root = root.replace('/', '.').replace('\\', '.')
        files = [i for i in files if '.md' in i]
        if len(files)==0:continue
        dic[root] = files
    return dic

if __name__ == '__main__':
    setup(name='planer-zoo',
        version='0.17',
        url='https://github.com/Image-Py/planer-zoo',
        description='toolbox of planer',
        long_description=descr,
        author='YXDragon',
        author_email='yxdragon@imagepy.org',
        license='BSD 3-clause',
        packages=find_packages(),
        package_data=get_data_files(),
        install_requires=[
            'planer',
            'tqdm'
        ],
    )
