import torch

def restore_checkpoint(ckpt_dir, state, device):
  
  loaded_state = torch.load(ckpt_dir, map_location=device)
  
  
  state['model'].load_state_dict(loaded_state['model'], strict=False)
  state['step'] = loaded_state['step']
 
  return state


def save_checkpoint(ckpt_dir, state):
  # 'ema': state['ema'].state_dict(),
  state_dict = state['model'].state_dict()
  
  keys_to_delete = [key for key in state_dict.keys() if ("module.lm_backbone" in key)]

  saved_state = {
    'model': state_dict,
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)