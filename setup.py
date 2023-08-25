from setuptools import setup, find_packages

setup(
    name='jaxlm',
    version='0.1',
    packages=find_packages(),
    author='Xingyao Wang',
    author_email='xingyao6@illinois.edu',
    description='A simplistic Jax-based Language Model Training and Serving Framework.',
    url='https://github.com/xingyaoww/jaxlm',
    license='Apache License 2.0',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
