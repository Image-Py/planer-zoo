from setuptools import setup, find_packages

descr = """Some useful models based on planer"""

if __name__ == '__main__':
    setup(name='planer-store',
        version='0.11',
        url='https://github.com/Image-Py/planer-store',
        description='toolbox of planer',
        long_description=descr,
        author='Y.Dong, YXDragon',
        author_email='yxdragon@imagepy.org',
        license='BSD 3-clause',
        packages=find_packages(),
        package_data={},
        install_requires=[
            'planer',
            'tqdm'
        ],
    )
