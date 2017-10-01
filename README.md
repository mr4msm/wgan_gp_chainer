This is an implementation of [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) in Chainer v3.0.0rc1

# Requirements
Chainer v3.0.0rc1, OpenCV, etc.  
The scripts work on Python 2.7.13.

# How to generate an image
```
python generate_image.py config.py -p example_food-101/trained-params_gen_update-000020000.npz
```
you can generate various images by changing the random_seed option
```
python generate_image.py config.py -p example_food-101/trained-params_gen_update-000020000.npz -r 1
```

# Dataset
I resized the images to 64x64 before training.
* [Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)
* [Birds](http://www-cvr.ai.uiuc.edu/ponce_grp/data/)  
