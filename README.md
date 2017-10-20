This is an implementation of [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) in [Chainer](https://github.com/chainer/chainer) v3.0.0.

# Requirements
Chainer v3.0.0, OpenCV, etc.  
The scripts work on Python 2.7.13 and 3.6.1.

# How to generate images
```
$ python generate_image.py example_food-101/config.py -p example_food-101/trained-params_gen_update-000040000.npz
```
You can generate various images by changing the random_seed option.
```
$ python generate_image.py example_food-101/config.py -r 1 -p example_food-101/trained-params_gen_update-000040000.npz
```

## Example Food-101
![example_image_food-101](https://raw.githubusercontent.com/mr4msm/wgan_gp_chainer/master/example_food-101/example_update-000040000.png)

## Example Birds
![example_image_birds](https://raw.githubusercontent.com/mr4msm/wgan_gp_chainer/master/example_birds/example_update-000040000.png)

# Dataset
I resized the images to 64x64 before training.
* [Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)  
Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc. Food-101 -- Mining Discriminative Components with Random Forests. European Conference on Computer Vision, 2014.
* [Birds](http://www-cvr.ai.uiuc.edu/ponce_grp/data/)  
Svetlana Lazebnik, Cordelia Schmid, and Jean Ponce. A Maximum Entropy Framework for Part-Based Texture and Object Recognition. Proceedings of the IEEE International Conference on Computer Vision, Beijing, China, October 2005, vol. 1, pp. 832-838.
