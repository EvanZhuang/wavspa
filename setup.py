
import setuptools 
  
with open("README.md", "r") as fh: 
    description = fh.read() 
  
setuptools.setup( 
    name="wavspa", 
    version="0.0.1", 
    author="Yufan Zhuang", 
    author_email="contact@gfg.com", 
    packages=["wavspa"], 
    description="Adaptive Wavelet Transform", 
    long_description=description, 
    long_description_content_type="text/markdown", 
    url="https://github.com/gituser/test-tackage", 
    license='MIT', 
    python_requires='>=3.8', 
    install_requires=[] 
)