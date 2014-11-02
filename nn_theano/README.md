# nn_theano

Theano is good, but I'm not quite comfortable with Pylearn2, so trying to write a easily configurable neural net library.

Here are some simple examples using nn_theano, more details can be found in [test.py](test.py).

```python
# different kinds of layers
# 'full': FullConnectLayer, connection layer
# 'sigm': SigmoidLayer, activation layer
# 'deno': DenoisingLayer, just for denoising autoencoders
# 'drop': DropoutLayer, activation layer
# 'relu': ReLULayer, activation layer

'''
Construct a neural net via `Net`, to which the first arg is a list of lists, specifying the architecture with inside lists being specifics of each layer.
For example,
    [['full', 100, {}], ['sigm', 50, {}]]
specifies that 1st layer is a FullConnectLayer with 100 input size, and 2nd layer is a SigmoidLayer with 50 input size, more layer-specific args can be specified inside `{}`.
Note that, `Net` will find out the weight matrix shape of connection layer by combining the input sizes of these 2 adjacent layers.
'''
# build a denoising autoencoder
dae = Net([['deno', 100, {'level': 0.2}],
           ['full', 100, {}],
           ['sigm', 50, {}],
           ['full', 50, {}],
           ['sigm', 100, {}]],  # architecture

          loss_type='ceml',  # cross entropy, multilabel Bernoulli
          updater_args={
            'base_lr': 0.1,
            'l2_decay': 0.0,
            'tune_lr_type': 0,
            'update_type': 0
          }  # parameters for sgd optimization
         )

'''
Since it's all based on theano, you should pack your data into tehano.shared variables before using them.
'''
# pack data into theano objects
raw_x = np.random.rand(50, 100)
raw_x_validate = np.random.rand(50, 100)

x = theano.shared(raw_x, borrow=True)  # pack
x_validate = theano.shared(raw_x_validate, borrow=True)

dae.train(x=x, y=x,   # training data
          x_v=x_validate, y_v=x_validate,  # validation data
          n_epochs=10, batch_size=20)

```

