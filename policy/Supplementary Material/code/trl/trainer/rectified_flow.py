from ..rectified_flow.new_update.config import get_config 
from ..rectified_flow.new_update.sampling import get_sampling_fn
from ..rectified_flow.new_update.sde_lib import RectifiedFlow 
from ..rectified_flow.new_update.utils import create_model 
from ..rectified_flow.new_update.util import restore_checkpoint 
from ..rectified_flow.new_update import network

class Rectified_Flow:
    def __init__(self):
        self.config = get_config()
        self.score_model = create_model(self.config)
        sampling_shape = (self.config.eval.batch_size,4)
        eval_sde = RectifiedFlow(init_type=self.config.sampling.init_type, noise_scale=self.config.sampling.init_noise_scale, \
            use_ode_sampler=self.config.sampling.use_ode_sampler, sigma_var=self.config.sampling.sigma_variance, \
                ode_tol=self.config.sampling.ode_tol, sample_N=self.config.sampling.sample_N)
        sampling_eps = 1e-3
        self.sampling_fn = get_sampling_fn(self.config, eval_sde, sampling_shape, None, sampling_eps)
        self.state = dict(model=self.score_model, step=0)
     
        ckpt_path = 'xxx/project/trl/trl/rectified_flow/log/4_pure_llama3_smac/checkpoints/checkpoint_xx.pth'
       
        self.state = restore_checkpoint(ckpt_path, self.state, device=self.config.device)

    def evaluate(self, data):
      
        samples, n = self.sampling_fn(self.state['model'], data)
        return samples
      
    