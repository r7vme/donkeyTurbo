from setuptools import setup

setup(name='donkeyturbo',
      version='0.1',
      description='Extension for donkeycar module',
      url='http://github.com/r7vme/donkeyTurbo',
      author='Roma Sokolkov',
      author_email='rsokolkov@gmail.com',
      license='MIT',
      packages=['donkeyturbo'],
      zip_safe=False,
      install_requires=[
          'donkeycar==2.2.4',
          'docopt',
          'pillow',
          'numpy',
          ])
