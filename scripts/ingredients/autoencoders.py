"""
Sacred Ingredient for Autoencoders

This ingredient takes an autoencoder configuration and produces an intance
of the specified generative model^*. it also has a convenience command used
to print the architecture.

This function will add the corresponding last layer to the model so that the
output of the encoder has the appropriate size. If the decoder layers are not
specified then it will try to transpose the definitions by itself^{**}.

To produce the models it uses the FeedForward module to stack the layers. This
is just a subclass of nn.Sequential with some added properties for convenience.
See 'src/models/feedforward.py' for more info.

^* Currently the only type fo model it supports is LGM
^{**} This might lead to an inconsistent definition where an unflatten layer is
applied **before** a ReLU activation instead of after. This has no functional
implications though. If you are feeling pedantic about these things you will
have to define your decoders explcitly.
"""


import sys
from sacred import Ingredient

if sys.path[0] != '../src':
    sys.path.insert(0, '../src')

from models.feedforward import FeedForward, transpose_layer_defs
from models.lgm import LGM, pLGM
from configs import cnnvae2

model = Ingredient('model')


# Print the model
@model.command(unobserved=True)
def show():
    model = init_lgm()
    print(model)


@model.capture
def init_lgm(gm_type, input_size, encoder_layers, latent_size,
             decoder_layers=None):
    
    encoder_layers += [('linear', [2 * latent_size])]
    encoder = FeedForward(input_size, encoder_layers, flatten=False)

    if decoder_layers is None:
        decoder_layers = encoder_layers[:-1]
        decoder_layers.append(('linear', [latent_size]))

        decoder_layers = transpose_layer_defs(decoder_layers, input_size)

    #decoder = FeedForward(latent_size, decoder_layers, flatten=True)
    decoder = FeedForward(latent_size, decoder_layers, flatten=False)

    return LGM(latent_size, encoder, decoder)




def init_plgm(encoder_name, decoder_name, input_size, latent_size):
    
    doubled_input_size = tuple([input_size[0]*2] + input_size[1:])
    list_encoder_name = ['higgins', 'burgess', 'burgess_v2', 'mpcnn', 'mathieu', 'kim']
    if encoder_name in list_encoder_name:
        encoder_layers = (getattr(cnnvae2, encoder_name).encoder_layers).copy()
        encoder_layers += [('linear', [latent_size])]
        encoder = FeedForward(doubled_input_size, encoder_layers, flatten=False)
    else:
        raise
    
    if decoder_name == 'CycleGAN':
        from models.networks_cyclegan import ResnetGenerator
        decoder = ResnetGenerator(input_size[0] + latent_size, input_size[0], output_activation = None)
    #decoder = FeedForward(latent_size + input_size[0]//2, decoder_layers, flatten=True)

    return pLGM(latent_size, encoder, decoder)


