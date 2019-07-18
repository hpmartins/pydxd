from setuptools import setup

def readme():
    with open('README') as f:
        return f.read()

setup(name='pyxro',
      version='0.1',
      description='Python YXRO',
      long_description=readme(),
      url='',
      author='H. P. Martins',
      author_email='hpmartins@lbl.gov',
      license='MIT',
      packages=['pyxro'],
      package_dir={'pyxro': 'pyxro'},
      #package_data={'esutils': ['data/parameters.json.gz']},
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
      ],
      include_package_data=True,
      zip_safe=False)
