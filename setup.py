from distutils.core import setup

with open("README.md", 'r') as fh:
    long_description = fh.read()

setup(
    name='quickthoughts',
    version='0.0.1',
    author='Jesse Cai',
    author_email='jcjessecai@gmail.com',
    packages=['quickthoughts'],
    url='https://github.com/jcaip/quickthoughts',
    license='LICENSE.txt',
    description='pytorch sentence vectors.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "torch >= 1.1.0",
        "gensim >= 3.4.0",
    ],
)
