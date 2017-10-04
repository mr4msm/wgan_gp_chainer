# -*- coding: utf-8 -*-
"""Train Critic and Generator using Wasserstein GAN with gradient penalty."""

import argparse
import chainer
import imp
import numpy as np
import os
from chainer import Variable, cuda, serializers, using_config
from chainer import computational_graph as graph
from chainer import functions as F
from batch_generator import ImageBatchGenerator
from datetime import datetime as dt

SAVE_PARAMS_FORMAT = 'trained-params_{0}_update-{1:09d}.npz'
SAVE_STATE_FORMAT = 'optimizer-state_{0}_update-{1:09d}.npz'


def load_module(module_path):
    """Load Python module."""
    head, tail = os.path.split(module_path)
    module_name = os.path.splitext(tail)[0]
    info = imp.find_module(module_name, [head])
    return imp.load_module(module_name, *info)


def parse_arguments():
    """Define and parse positional/optional arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'config',
        help='a configuration file in which networks, optimizers and hyper params are defined (.py)'
    )
    parser.add_argument(
        'dataset',
        help='a text file in which image files are listed'
    )
    parser.add_argument(
        '-c', '--computational_graph', action='store_true',
        help='if specified, build computational graph'
    )
    parser.add_argument(
        '-g', '--gpu', default=-1, type=int,
        help='GPU ID (negative value indicates CPU)'
    )
    parser.add_argument(
        '-o', '--output', default='./',
        help='a directory in which output files will be stored'
    )
    parser.add_argument(
        '-p', '--param_cri', default=None,
        help='trained parameters for Critic saved as serialized array (.npz | .h5)'
    )
    parser.add_argument(
        '-P', '--param_gen', default=None,
        help='trained parameters for Generator saved as serialized array (.npz | .h5)'
    )
    parser.add_argument(
        '-s', '--state_cri', default=None,
        help='optimizer state for Critic saved as serialized array (.npz | .h5)'
    )
    parser.add_argument(
        '-S', '--state_gen', default=None,
        help='optimizer state for Generator saved as serialized array (.npz | .h5)'
    )

    return parser.parse_args()


def initialize_parameter(model, param, output_path):
    """Save initial params or load params to resume."""
    if param is None:
        serializers.save_npz(output_path, model)
        print('save ' + output_path)
    else:
        ext = os.path.splitext(param)[1]

        if ext == '.npz':
            load_func = serializers.load_npz
        elif ext == '.h5':
            load_func = serializers.load_hdf5
        else:
            raise TypeError(
                'The format of \"{}\" is not supported.'.format(param))

        load_func(param, model)
        print('load ' + param)


def initialize_optimizer(optimizer, state, output_path):
    """Save initial state or load state to resume."""
    initialize_parameter(optimizer, state, output_path)


def train_wgan_gp():
    """Train Critic and Generator using Wasserstein GAN with gradient penalty."""
    # parse arguments
    args = parse_arguments()
    config = load_module(args.config)
    out_dir = args.output
    gpu_id = args.gpu

    # make output directory, if needed
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print('mkdir ' + out_dir)
    assert os.path.isdir(out_dir)

    # setup network model, optimizer, and constant values to control training
    z_vec_dim = config.Z_VECTOR_DIM
    batch_size = config.BATCH_SIZE
    update_max = config.UPDATE_MAX
    update_save_params = config.UPDATE_SAVE_PARAMS
    update_cri_per_gen = getattr(config, 'UPDATE_CRI_PER_GEN', 5)
    gp_lambda = getattr(config, 'LAMBDA', 10)

    model_gen = config.Generator()
    optimizer_gen = config.OPTIMIZER_GEN
    optimizer_gen.setup(model_gen)

    model_cri = config.Critic()
    optimizer_cri = config.OPTIMIZER_CRI
    optimizer_cri.setup(model_cri)

    # setup batch generator
    with open(args.dataset, 'r') as f:
        input_files = [line.strip() for line in f.readlines()]
    batch_generator = ImageBatchGenerator(input_files, batch_size,
                                          config.HEIGHT, config.WIDTH,
                                          channel=config.CHANNEL,
                                          shuffle=True,
                                          flip_h=getattr(config, 'FLIP_H', False))
    sample_num = batch_generator.n_samples

    # show some settings
    print('sample num = {}'.format(sample_num))
    print('mini-batch size = {}'.format(batch_size))
    print('max update count = {}'.format(update_max))
    print('updates per saving params = {}'.format(update_save_params))
    print('critic updates per generator update = {}'.format(update_cri_per_gen))
    print('lambda for gradient penalty = {}'.format(gp_lambda))

    # save or load initial parameters for Critic
    initialize_parameter(model_cri, args.param_cri,
                         os.path.join(out_dir, SAVE_PARAMS_FORMAT.format('cri', optimizer_cri.t)))
    # save or load initial parameters for Generator
    initialize_parameter(model_gen, args.param_gen,
                         os.path.join(out_dir, SAVE_PARAMS_FORMAT.format('gen', optimizer_gen.t)))
    # save or load initial optimizer state for Critic
    initialize_optimizer(optimizer_cri, args.state_cri,
                         os.path.join(out_dir, SAVE_STATE_FORMAT.format('cri', optimizer_cri.t)))
    # save or load initial optimizer state for Generator
    initialize_optimizer(optimizer_gen, args.state_gen,
                         os.path.join(out_dir, SAVE_STATE_FORMAT.format('gen', optimizer_gen.t)))

    # set current device and copy model to it
    xp = np
    if gpu_id >= 0:
        cuda.get_device(gpu_id).use()
        model_gen.to_gpu(device=gpu_id)
        model_cri.to_gpu(device=gpu_id)
        xp = cuda.cupy

    # set global configuration
    chainer.global_config.enable_backprop = True
    chainer.global_config.enable_double_backprop = True
    chainer.global_config.train = True

    print('** chainer global configuration **')
    chainer.global_config.show()

    initial_t = optimizer_gen.t
    sum_count = 0
    sum_loss_g = 0.
    sum_loss_c = 0.

    # training loop
    while optimizer_gen.t < update_max:

        for _ in range(update_cri_per_gen):
            x_t = batch_generator.next()
            if gpu_id >= 0:
                x_t = cuda.to_gpu(x_t, device=gpu_id)

            z = xp.random.normal(loc=0., scale=1.,
                                 size=(batch_size, z_vec_dim)).astype('float32')

            with using_config('enable_backprop', False):
                x_g = model_gen(z).data

            y_g = model_cri(x_g)
            y_t = model_cri(x_t)
            loss_c = F.sum(y_g) - F.sum(y_t)

            # calculate gradient penalty
            differences = x_t - x_g
            alpha = xp.random.uniform(low=0., high=1.,
                                      size=((batch_size,) +
                                            (1,) * (differences.ndim - 1))
                                      ).astype('float32')
            interpolates = Variable(x_t - alpha * differences)

            gradients = chainer.grad([model_cri(interpolates)],
                                     [interpolates],
                                     enable_double_backprop=True)[0]

            if gradients.data.ndim > 1:
                slopes = F.sqrt(F.sum(gradients * gradients,
                                      axis=tuple(range(1, gradients.data.ndim))))
            else:
                slopes = gradients

            gradient_penalty = gp_lambda * F.sum((slopes - 1.) * (slopes - 1.))

            loss_c += gradient_penalty
            loss_c /= batch_size

            # update Critic
            model_cri.cleargrads()
            loss_c.backward()
            optimizer_cri.update()

            sum_loss_c += float(loss_c.data) / update_cri_per_gen

        z = xp.random.normal(loc=0., scale=1.,
                             size=(batch_size, z_vec_dim)).astype('float32')
        x_g = model_gen(z)
        y_g = model_cri(x_g)
        loss_g = -F.sum(y_g) / batch_size

        # update Generator
        model_cri.cleargrads()
        model_gen.cleargrads()
        loss_g.backward()
        optimizer_gen.update()

        sum_loss_g += float(loss_g.data)
        sum_count += 1

        # show losses
        print('{0}: update # {1:09d}: C loss = {2:6.4e}, G loss = {3:6.4e}'.format(
            str(dt.now()), optimizer_gen.t,
            float(loss_c.data), float(loss_g.data)))

        # output computational graph, if needed
        if args.computational_graph and optimizer_gen.t == (initial_t + 1):
            with open('graph.dot', 'w') as o:
                o.write(graph.build_computational_graph((loss_c, )).dump())
            print('graph generated')

        # show mean losses, save interim trained parameters and optimizer states
        if optimizer_gen.t % update_save_params == 0:
            print('{0}: mean of latest {1:06d} in {2:09d} updates : C loss = {3:7.5e}, G loss = {4:7.5e}'.format(
                str(dt.now()), sum_count, optimizer_gen.t, sum_loss_c / sum_count, sum_loss_g / sum_count))
            sum_count = 0
            sum_loss_g = 0.
            sum_loss_c = 0.

            output_file_path = os.path.join(
                out_dir, SAVE_PARAMS_FORMAT.format('gen', optimizer_gen.t))
            serializers.save_npz(output_file_path, model_gen)
            print('save ' + output_file_path)

            output_file_path = os.path.join(
                out_dir, SAVE_PARAMS_FORMAT.format('cri', optimizer_cri.t))
            serializers.save_npz(output_file_path, model_cri)
            print('save ' + output_file_path)

            output_file_path = os.path.join(
                out_dir, SAVE_STATE_FORMAT.format('gen', optimizer_gen.t))
            serializers.save_npz(output_file_path, optimizer_gen)
            print('save ' + output_file_path)

            output_file_path = os.path.join(
                out_dir, SAVE_STATE_FORMAT.format('cri', optimizer_cri.t))
            serializers.save_npz(output_file_path, optimizer_cri)
            print('save ' + output_file_path)

    # save final trained parameters and optimizer states
    output_file_path = os.path.join(
        out_dir, SAVE_PARAMS_FORMAT.format('gen', optimizer_gen.t))
    serializers.save_npz(output_file_path, model_gen)
    print('save ' + output_file_path)

    output_file_path = os.path.join(
        out_dir, SAVE_PARAMS_FORMAT.format('cri', optimizer_cri.t))
    serializers.save_npz(output_file_path, model_cri)
    print('save ' + output_file_path)

    output_file_path = os.path.join(
        out_dir, SAVE_STATE_FORMAT.format('gen', optimizer_gen.t))
    serializers.save_npz(output_file_path, optimizer_gen)
    print('save ' + output_file_path)

    output_file_path = os.path.join(
        out_dir, SAVE_STATE_FORMAT.format('cri', optimizer_cri.t))
    serializers.save_npz(output_file_path, optimizer_cri)
    print('save ' + output_file_path)


if __name__ == '__main__':
    train_wgan_gp()
