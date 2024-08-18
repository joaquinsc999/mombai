from setuptools import setup, find_packages

setup(
    name="mombai",
    version="0.1.0",
    description="A deep learning library for advanced neural network layers.",
    author="Joaquín Solórzano",
    author_email="tu_correo@example.com",
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
