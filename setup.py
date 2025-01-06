from setuptools import setup, find_packages

setup(
    name="mombai",
    version="1.0.0",
    description="A deep learning library for advanced neural network layers.",
    include_package_data=True,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Joaquín Francisco Solórzano Corea',
    author_email='joaquinscorea@gmail.com',
    url="https://github.com/joaquinsc999/mombai",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.0",
        # otros requerimientos
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
