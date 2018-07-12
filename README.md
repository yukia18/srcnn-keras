# srcnn_keras
## Abstruct
This is an implementation of SRCNN using keras. You can conduct simple experiments of super resolution on Set5 or your own images.
## Requirements
- python 3.6
- keras
- opencv
## Usage
### Train
`python train.py`  
To conduct using GPU is recommended. There is a pretrained model in './model' in my environment, so you don't have to train model.
### Test
`python test.py`  
Outputs are restored in './result'. If you want to try super resolution to your own images, please put them in './test'.
- original: images preprocessed as downsampling 1/scale (default scale=3)  
- answer: target images (raw images in './test')
- input: 'original' images preprocessed as upsampling scale/1 (default scale=3)
- predicted: outputs images
## References
[kweisamx/TensorFlow-SRCNN](https://github.com/kweisamx/TensorFlow-SRCNN)
