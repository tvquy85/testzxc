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

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time

import numpy as np

import logging

import losses
import sampling
import utils as mutils

import datasets
from torch.utils.data import DataLoader, Dataset

import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from util import save_checkpoint, restore_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
import network
import os 
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

FLAGS = flags.FLAGS

class PTDataset(Dataset):
    def __init__(self, file_path, device = "cuda:1"):
        
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        data = torch.load(file_path, map_location=device)
        self.attention_mask = data["attention_mask"]
        self.last_layer_hidden_states = data["last_layer_hidden_states"]
        self.gt_action = data["gt_action"]
        self.action_prob = data["action_prob"]
        self.position_embeddings = data["position_embeddings"]

    def __len__(self):
        
        
        return len(self.attention_mask)

    def __getitem__(self, idx):
        
        return self.attention_mask[idx], self.last_layer_hidden_states[idx], self.gt_action[idx], self.action_prob[idx], self.position_embeddings[idx]

class NPTDataset(Dataset):
    def __init__(self, file_path, device = "cuda:1"):
       
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        data = torch.load(file_path, map_location=device)
       
        self.attention_mask = data["attention_mask"]
        self.last_layer_hidden_states = data["last_layer_hidden_states"]
        self.gt_action = data["gt_action"]
        # self.prompt = data["prompt"]
        # self.action_prob = data["action_prob"]
       
        self.position_embeddings = data["position_embeddings"]

    def __len__(self):
        
        
        return len(self.gt_action)

    def __getitem__(self, idx):
        
        return self.attention_mask[idx], self.last_layer_hidden_states[idx], self.gt_action[idx], self.position_embeddings[idx]


def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs


  tb_dir = os.path.join(workdir, "tensorboard")

  writer = tensorboard.SummaryWriter(tb_dir)
  
  eval_dir = os.path.join(workdir, "eval")
  eval_writer = tensorboard.SummaryWriter(eval_dir)
  # Initialize model.
  score_model = mutils.create_model(config)

  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, step=0)#, ema=ema

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments

  if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

  # Resume training when intermediate checkpoints are detected
  initial_step = int(state['step'])


  

# 实例化 Dataset 和 DataLoader

  # 
  train_ds = PTDataset("xxx/dataset/tulu2/mmlu/llama3_base_train.pt",device = "cuda:1")
  eval_ds = NPTDataset("xxx/dataset/tulu2/mmlu/llama3_base_eval.pt",device = "cuda:2") 
  eval_negative_ds = NPTDataset("xxx/dataset/tulu2/mmlu/llama3_test_negative_mmlu.pt",device = "cuda:3")
  eval_true_negative_ds = NPTDataset("xxx/dataset/tulu2/mmlu/llama3_test_true_negative_mmlu.pt",device = "cuda:3")
  eval_false_negative_ds = NPTDataset("xxx/dataset/tulu2/mmlu/llama3_test_false_negative_mmlu.pt",device = "cuda:3")
 
  train_ds_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, drop_last=True)
  eval_ds_loader = torch.utils.data.DataLoader(eval_ds, batch_size=config.eval.batch_size, shuffle=True, drop_last=True)
  
  eval_negative_ds_loader = torch.utils.data.DataLoader(eval_negative_ds, batch_size=config.eval.batch_size, shuffle=True, drop_last=True)
  eval_true_negative_ds_loader = torch.utils.data.DataLoader(eval_true_negative_ds, batch_size=config.eval.batch_size, shuffle=True, drop_last=True)
  eval_false_negative_ds_loader = torch.utils.data.DataLoader(eval_false_negative_ds, batch_size=config.eval.batch_size, shuffle=True, drop_last=True)

  # Setup SDEs
  if config.training.sde.lower() == 'rectified_flow':
    sde = sde_lib.RectifiedFlow(init_type=config.sampling.init_type, noise_scale=config.sampling.init_noise_scale, use_ode_sampler=config.sampling.use_ode_sampler)
    eval_sde = sde_lib.RectifiedFlow(init_type=config.sampling.init_type, noise_scale=config.sampling.init_noise_scale, use_ode_sampler=config.sampling.use_ode_sampler, sigma_var=config.sampling.sigma_variance, ode_tol=config.sampling.ode_tol, sample_N=config.sampling.sample_N)
    sampling_eps = 1e-3
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)

  if (config.eval.enable_sampling) or (config.eval.enable_figures_only):
   
    sampling_shape = (config.eval.batch_size,12) #12 for samc, 4 for mmlu, 5 for mathqa
                      
    sampling_fn = sampling.get_sampling_fn(config, eval_sde, sampling_shape, None, sampling_eps)


  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))

  step = initial_step - 1
  print('start')
  while step <= (num_train_steps + 1):
    for _, (attention_mask, last_layer_hidden_states, gt_action, action_prob, position_embeddings) in enumerate(train_ds_loader): 
      
        attention_mask = attention_mask.to(config.device)
        last_layer_hidden_states = last_layer_hidden_states.to(config.device)
        gt_action = gt_action.to(config.device)
       
        action_prob = torch.softmax(action_prob[:,:,0], dim=-1).to(config.device)
        position_embeddings = position_embeddings.to(config.device)
        
      
        step += 1
        
        data = (attention_mask, last_layer_hidden_states, gt_action, action_prob, position_embeddings, False)
        # Execute one training step
        
        loss, p_accuracy, rp_accuracy, rn_accuracy, accuracy= train_step_fn(state, data)
        print(f"step:{step}, training_loss:{loss}")
        print(f"training_accuracy:{accuracy}")
        if step % config.training.log_freq == 0:
          logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
          writer.add_scalar("training_loss", loss, step)
          writer.add_scalar("training_accuracy", accuracy, step)
   
        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
          episode_evaluate(config, score_model, eval_ds_loader, eval_writer, sampling_fn, step, True, "eval")
         
        if step % config.training.eval_freq == 0:
          episode_evaluate(config, score_model, eval_negative_ds_loader, eval_writer, sampling_fn, step, True, "eval_negative")
        
        if step % config.training.eval_freq == 0:
          episode_evaluate(config, score_model, eval_true_negative_ds_loader, eval_writer, sampling_fn, step, True, "eval_true_negative")
        
        if step % config.training.eval_freq == 0:
          episode_evaluate(config, score_model, eval_false_negative_ds_loader, eval_writer, sampling_fn, step, True, "eval_false_negative")
        
     
        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
          # Save the checkpoint.
          save_step = step // config.training.snapshot_freq
          save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)


def episode_evaluate(config, score_model, eval_ds_loader, writer, sampling_fn, step, negative, name):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  

  # Create the one-step evaluation function when loss computation is enabled
  if config.eval.enable_loss:
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting

    reduce_mean = config.training.reduce_mean
    eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                   reduce_mean=reduce_mean,
                                   continuous=continuous,
                                   likelihood_weighting=likelihood_weighting)


  
  # Build the likelihood computation function when likelihood is enabled
  if config.eval.enable_bpd:
    likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

  # Build the sampling function when sampling is enable
    
  # Compute the loss function on the full evaluation dataset if loss computation is enabled
  if config.eval.enable_loss:
    all_losses = []
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
    for i, batch in enumerate(eval_iter):
      eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
      eval_batch = eval_batch.permute(0, 3, 1, 2)
      eval_batch = scaler(eval_batch)
      eval_loss = eval_step(state, eval_batch)
      all_losses.append(eval_loss.item())
      if (i + 1) % 1000 == 0:
        logging.info("Finished %dth step loss evaluation" % (i + 1))

    # Save loss values to disk or Google Cloud Storage
    all_losses = np.asarray(all_losses)
    with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
      fout.write(io_buffer.getvalue())

  # Compute log-likelihoods (bits/dim) if enabled
  if config.eval.enable_bpd:
    bpds = []
    for repeat in range(bpd_num_repeats):
      bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
      for batch_id in range(len(ds_bpd)):
        batch = next(bpd_iter)
        eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
        eval_batch = eval_batch.permute(0, 3, 1, 2)
        eval_batch = scaler(eval_batch)
        bpd = likelihood_fn(score_model, eval_batch)[0]
        bpd = bpd.detach().cpu().numpy().reshape(-1)
        bpds.extend(bpd)
        logging.info(
          "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
        bpd_round_id = batch_id + len(ds_bpd) * repeat
        # Save bits/dim to disk or Google Cloud Storage
        with tf.io.gfile.GFile(os.path.join(eval_dir,
                                            f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                                "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, bpd)
          fout.write(io_buffer.getvalue())
  
  # Generate samples and compute IS/FID/KID when enabled
  if config.eval.enable_sampling:
   
    accuracy = []
    if negative:
      for _, (attention_mask, last_layer_hidden_states, gt_action, position_embeddings) in enumerate(eval_ds_loader):
        break
    else:
      for _, (attention_mask, last_layer_hidden_states, gt_action, action_prob, position_embeddings) in enumerate(eval_ds_loader):
        break
    attention_mask = attention_mask.to(config.device)
    last_layer_hidden_states= last_layer_hidden_states.to(config.device)
    gt_action= gt_action.to(config.device)
    position_embeddings= position_embeddings.to(config.device)
    data = (attention_mask, last_layer_hidden_states, position_embeddings)#
    for _ in range(3):
     
      samples, n = sampling_fn(score_model, data)
      accuracy.append((torch.argmax(samples,dim=-1)==gt_action).sum()/gt_action.shape[0])
    print(f'{name}_accuracy:{torch.tensor(accuracy).mean()}')
    writer.add_scalar(f"{name}_accuracy", torch.tensor(accuracy).mean().item(), step)
    

def evaluate(config,
             workdir,
             eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  writer = tensorboard.SummaryWriter(eval_dir)
 
  tokenizer = AutoTokenizer.from_pretrained('xxx/model', legacy=False)
  tokenizer.bos_token_id = 1
  tokenizer.pad_token = tokenizer.eos_token
  stop_token_ids = [0]
 
  eval_ds = PTDataset("xxx/dataset/smac_dataset_test_rectified_v2.pt",device = "cpu")
  eval_ds_loader = torch.utils.data.DataLoader(eval_ds, batch_size=config.eval.mini_batch_size, shuffle=False, drop_last=True)
  

  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  
  state = dict(optimizer=optimizer, model=score_model, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'rectified_flow':
    sde = sde_lib.RectifiedFlow(init_type=config.sampling.init_type, noise_scale=config.sampling.init_noise_scale, use_ode_sampler=config.sampling.use_ode_sampler, sigma_var=config.sampling.sigma_variance, ode_tol=config.sampling.ode_tol, sample_N=config.sampling.sample_N)
    sampling_eps = 1e-3
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Create the one-step evaluation function when loss computation is enabled
  if config.eval.enable_loss:
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting

    reduce_mean = config.training.reduce_mean
    eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                   reduce_mean=reduce_mean,
                                   continuous=continuous,
                                   likelihood_weighting=likelihood_weighting)

  # Build the likelihood computation function when likelihood is enabled
  if config.eval.enable_bpd:
    likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

  # Build the sampling function when sampling is enabled
  if (config.eval.enable_sampling) or (config.eval.enable_figures_only):
    mini_sampling_shape = (config.eval.batch_size,12)
    sampling_shape = (config.eval.batch_size,12)
                     
    mini_sampling_fn = sampling.get_sampling_fn(config, sde, mini_sampling_shape, None, sampling_eps)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, None, sampling_eps)


  begin_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  step = 0
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
  
    
   
    
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
    state = restore_checkpoint(ckpt_path, state, device=config.device)
     
    
    # Compute the loss function on the full evaluation dataset if loss computation is enabled
    if config.eval.enable_loss:
      all_losses = []
      eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
      for i, batch in enumerate(eval_iter):
        eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
        eval_batch = eval_batch.permute(0, 3, 1, 2)
        eval_batch = scaler(eval_batch)
        eval_loss = eval_step(state, eval_batch)
        all_losses.append(eval_loss.item())
        if (i + 1) % 1000 == 0:
          logging.info("Finished %dth step loss evaluation" % (i + 1))

      # Save loss values to disk or Google Cloud Storage
      all_losses = np.asarray(all_losses)
      with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
        fout.write(io_buffer.getvalue())

    # Compute log-likelihoods (bits/dim) if enabled
    if config.eval.enable_bpd:
      bpds = []
      for repeat in range(bpd_num_repeats):
        bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
        for batch_id in range(len(ds_bpd)):
          batch = next(bpd_iter)
          eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
          eval_batch = eval_batch.permute(0, 3, 1, 2)
          eval_batch = scaler(eval_batch)
          bpd = likelihood_fn(score_model, eval_batch)[0]
          bpd = bpd.detach().cpu().numpy().reshape(-1)
          bpds.extend(bpd)
          logging.info(
            "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
          bpd_round_id = batch_id + len(ds_bpd) * repeat
          # Save bits/dim to disk or Google Cloud Storage
          with tf.io.gfile.GFile(os.path.join(eval_dir,
                                              f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                                 "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, bpd)
            fout.write(io_buffer.getvalue())
    
    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:
      
      accuracy = []
  
      for r, (attention_mask, last_layer_hidden_states, gt_action, _, position_embeddings) in enumerate(eval_ds_loader): 
        attention_mask = attention_mask.to(config.device)
        last_layer_hidden_states= last_layer_hidden_states.to(config.device)
        gt_action= gt_action.to(config.device)
        position_embeddings = position_embeddings.to(config.device)
        data = (attention_mask, last_layer_hidden_states, position_embeddings)
     
        
        logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))
       
      
        samples, n = sampling_fn(score_model, data)
       
        accuracy.append(((torch.argmax(samples,dim=-1)==gt_action).sum()/gt_action.shape[0]).cpu())
        print(f'eval_accuracy:{(torch.argmax(samples,dim=-1)==gt_action).sum()/gt_action.shape[0]}')
        print(np.array(accuracy).sum()/len(accuracy))
        print(np.var(np.array(accuracy)))
    

if __name__ == "__main__":
  import config
  c = config.get_config()
  train(c,'xxx/project/trl/trl/rectified_flow/log/4_pure_llama3_base_mmlu')
  #evaluate(c,'xxx/project/trl/trl/rectified_flow/log/4_pure_llama3_smac_v2/')