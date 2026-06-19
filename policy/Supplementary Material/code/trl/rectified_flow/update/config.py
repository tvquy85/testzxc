# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training Rectified Flow on CIFAR-10 with DDPM++."""

# from configs.default_cifar10_configs import get_default_configs

import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 32
  config.training.rewrite_positive_batch_size = 0
  config.training.rewrite_negative_batch_size = 0
  training.n_iters = 1300001
  training.snapshot_freq = 100000
  training.log_freq = 50
  training.eval_freq = 200
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 10000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16
  
  sampling.sigma_variance = 0.0 # NOTE: sigma variance for turning ODE to SDE
  sampling.init_noise_scale = 1.0
  sampling.use_ode_sampler = 'euler'
  sampling.ode_tol = 1e-5
  sampling.sample_N = 10

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 11
  evaluate.end_ckpt = 11
  evaluate.mini_batch_size = 1
  evaluate.batch_size = 16
  evaluate.enable_sampling = True
  evaluate.num_samples = 1000
  evaluate.enable_loss = False
  evaluate.enable_bpd = False
  evaluate.enable_figures_only = False
  evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'CIFAR10'
  data.image_size = 32
  data.random_flip = True
  data.centered = False
  data.uniform_dequantization = False
  data.num_channels = 3

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_min = 0.01
  model.sigma_max = 50
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0.
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config

def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = 'rectified_flow'
    training.continuous = False
    training.snapshot_freq = 20000
    training.reduce_mean = True

    # sampling
    sampling = config.sampling
    sampling.method = 'rectified_flow'
    sampling.init_type = 'gaussian' 
    sampling.init_noise_scale = 1.0
    sampling.use_ode_sampler = 'euler' ### rk45 or euler
    sampling.ode_tol = 1e-5

    # data
    data = config.data
    data.centered = True

    # model
    model = config.model
    model.name = 'ncsnpp'
    model.scale_by_sigma = False
    model.ema_rate = 0.999999
    model.dropout = 0.15
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'relu'
    model.nf = 4096#llama3584 #qwen#  2304 #gemma
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 4
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = False
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'none'
    model.progressive_input = 'none'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.
    model.embedding_type = 'positional'
    model.fourier_scale = 16
    model.conv_size = 3

    return config


