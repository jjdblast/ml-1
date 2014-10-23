machine-learning
================

Migrate some of machine learning code from Dropbox to github, maybe also trying to write some useful packages.

# nn
---
* *ae.py*
Contains sparse autoencoder (*SAE*) and denoising autoencoder (*DAE*).

*SAE* is implemented with *lbfgs* as the training method, whereas *DAE* is trained with *sgd* (constant learning rate). They are easy to use (I think so ;P), you just specify the size of input and hidden layers and call *train* to start the training, I have some demo code in [test.py](nn/test.py) on MNIST dataset, here are some of the results of image filters.

Denoising autoencoders with *corruption level=0.0*:

![alt text](nn/pic/dae_filter_level_0.png)

Denoising autoencoders with *corruption level=0.3*:

![alt text](nn/pic/dae_filter_level_30.png)

Denoising autoencoders with *corruption level=0.3* and *untied weightc*:

![alt text](nn/pic/dae_filter_level_30_untied.png)
