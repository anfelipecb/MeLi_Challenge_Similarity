from setuptools import setup, find_packages

setup(name="similarity",
      version="0.1",
      packages=find_packages(),
      install_requires=[
         "numpy", 
         "pandas", 
         "requests", 
         "matplotlib",
      ],
    )

