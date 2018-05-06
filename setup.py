from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name='fullnet_alchemist',
    version='0.1.0',
    url='https://github.com/smartcat-labs/fullnet_alchemist',
    license='',
    author='Stanko Kuveljic',
    author_email='stankokuveljic@gmail.com',
    description='Deep learning models in TensorFlow',
    long_description=readme,
    packages=find_packages(exclude=('training', 'tests', 'docs'))
)
