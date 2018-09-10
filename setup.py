from setuptools import setup

setup(name='uib_vfeatures',
      version='0.1',
      description='Vision features of generalistic use',
      url='http://github.com/storborg/funniest',
      author='UIB-UGIVIA',
      author_email='miquel.miro1@estudiants.uib.cat',
      license='MIT',
      packages=['uib_vfeatures'],
      install_requires=[
          'scikit-image',
          'opencv-python',
          'matplotlib',
          'scipy',
          'scikit-image',
          'numpy'
      ],
      zip_safe=False)
