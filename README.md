# camAIra Web Interface for SISR and SID
Application of Deep Learning Network for Enhance Low Resolution into High Resolution and Restoring Dark Images into Bright and Clear Image. Single Image Super-Resolution using ESRGAN as neural network to enhance Low-Res image into High-res Image. Using FCNN as a neural network for restoring dark images. Deep learning using Tensorflow, Pytorch, and ISR module as a framework model. For deploying model to website, using Flask and Flask Ngrok. Namely camAIra as Image Processing using Artificial Intelligence Technology.

The goal of this project is to upscale and improve the quality of low resolution images, restoring dark image into bright and clear image, and deploy models into web inteface.

This project contains Keras implementations of different Residual Dense Networks for Single Image Super-Resolution (ISR) as well as scripts to train these networks using content and adversarial loss components from Idealo. Framework using Flask and Flask Ngrok to deploy model as back-end. Using HTML and CSS to build front-end web page.

## Usage
### Instalation
- Install ISR from PyPI (recommended):
```
pip install ISR
```
- Install ISR from the GitHub source:
```
git clone https://github.com/idealo/image-super-resolution
cd image-super-resolution
python setup.py install
```
- Install Environment Needed by conda or pip
- Or Run install command in flask.ipynb 
- Run flask.ipynb to using public URL by Flask Ngrok framework
- Click at ngrok.io URL to see the result
