import utils, layers
import torch.nn as nn
import torch
import functools
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
default_initializer = layers.default_init
import gc
#from . 
get_act = layers.get_act

@utils.register_model(name='ncsnpp')
class Network(nn.Module):    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device
        #
         
        self.act = act = get_act(config)
        self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

        self.nf = nf = config.model.nf
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

        self.conditional = conditional = config.model.conditional  # noise-conditional
        fir = config.model.fir
        fir_kernel = config.model.fir_kernel
        self.skip_rescale = skip_rescale = config.model.skip_rescale
        self.resblock_type = resblock_type = config.model.resblock_type.lower()
        self.progressive = progressive = config.model.progressive.lower()
        self.progressive_input = progressive_input = config.model.progressive_input.lower()
        self.embedding_type = embedding_type = config.model.embedding_type.lower()
        
        init_scale = config.model.init_scale
        
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        assert embedding_type in ['fourier', 'positional']
       
        self.model2 = AutoModelForCausalLM.from_pretrained('/mnt/public/xxx/model/llama3.1-8B/Llama-3.1-8B-Instruct', torch_dtype=torch.bfloat16).cuda()
    
       
        modules = []
       
        
        self.embed_dim = 128


        
        modules.append(nn.Linear(self.embed_dim, nf))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)
        modules.append(nn.LayerNorm(normalized_shape=nf))
       
        modules.append(nn.Linear(12, nf)) #12 for samc, 4 for mmlu, 5 for mathqa
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)
        modules.append(nn.LayerNorm(normalized_shape=nf))
       
        
        self.lm_backbone = getattr(self.model2, self.model2.base_model_prefix)

       
        modules.append(self.lm_backbone.layers[-1])
        
        

        modules.append(nn.Linear(nf*2, nf))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)
        modules.append(nn.LayerNorm(normalized_shape=nf))
        modules.append(nn.Linear(nf*2, nf))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)
        modules.append(nn.LayerNorm(normalized_shape=nf))
        
        
        modules.append(nn.Linear(nf*2, nf))
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)
        modules.append(nn.LayerNorm(normalized_shape=nf))
        modules.append(nn.Linear(nf*2, 12)) #12 for samc, 4 for mmlu, 5 for mathqa
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)
        
        self.all_modules = nn.ModuleList(modules)

   
    
    def forward(self, data):
        
        # timestep/noise_level embedding; only for continuous training
        x, time_cond, hidden_states, attention_mask, position_embeddings = data #
       
       
        modules = self.all_modules
        m_idx = 0
        
        # Sinusoidal positional embeddings.
        timesteps = time_cond
        used_sigmas = self.sigmas[time_cond.long()]
        temb = layers.get_timestep_embedding(timesteps, self.embed_dim)

        for k in range(1):
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](temb)
            m_idx += 1
            if k < 1:
                temb = self.act(temb)
       
        b, d = temb.shape
        temb = temb.reshape(b,1,d)
        
        for k in range(1):
            x = modules[m_idx](x)
            m_idx += 1
            x = modules[m_idx](x)
            m_idx += 1
            if k < 1:
                x = self.act(x)
        b, d = x.shape
        x = x.reshape(b,1,d)
        
        with torch.no_grad():
            hidden_states = hidden_states.reshape(hidden_states.shape[0],-1,4096) # 4096 for llama3
            attention_mask = attention_mask.reshape(attention_mask.shape[0],-1)
            position_ids = attention_mask.cumsum(1) - attention_mask.long()
            hidden_states = torch.cat((hidden_states,x.bfloat16(),temb.bfloat16(),),dim=1)
            
            cache_position = torch.arange(
                    0,  hidden_states.shape[1], device=hidden_states.device
                )
            
            causal_mask = self.lm_backbone._update_causal_mask(
                    attention_mask, hidden_states, cache_position, None, False
                )
            
            (h,) = modules[m_idx](
                    hidden_states,
                    attention_mask=causal_mask,
                    past_key_value=None,
                    position_ids=position_ids,
                    output_attentions=False,
                    position_embeddings=(position_embeddings[:,0,0],position_embeddings[:,1,0]),#position_embeddings,#
                    cache_position=cache_position,
                    use_cache=False,  # otherwise mistral-based RM would error out
                    last_flag = True
                
                )
           
            h = h[:,-1].float()
            
       
        m_idx += 1
        
      
        h = torch.cat((h,x[:,0]),dim=-1)
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = self.act(h)
        h = torch.cat((h,temb[:,0]),dim=-1)
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = self.act(h)

        h = torch.cat((h,x[:,0]),dim=-1)
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = self.act(h)
        h = torch.cat((h,temb[:,0]),dim=-1)
        h = modules[m_idx](h)
        
        del x,temb
        
       
        return h
    
    