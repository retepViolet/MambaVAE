import torch
from torch import Tensor, nn
from torch.nn import functional as F
import transformers
from transformers.models.mamba import modeling_mamba

# import importlib
# import selective_scan_interface
# import selective_state_update
# importlib.reload(selective_scan_interface)
# importlib.reload(selective_state_update)

from .selective_scan_interface import mamba_inner_fn, selective_scan_fn
from .selective_state_update import selective_state_update


class MambaMixer(modeling_mamba.MambaMixer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_ssm_states: modeling_mamba.Optional[torch.Tensor] = None,
        cache_params: modeling_mamba.Optional[modeling_mamba.MambaCache] = None,
        cache_position: modeling_mamba.Optional[torch.LongTensor] = None,
        attention_mask: modeling_mamba.Optional[torch.LongTensor] = None,
    ):
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states).transpose(1, 2)
        ssm_state = None
        if self.training and cache_params is None:  # Doesn't support outputting the states -> used for training
            contextualized_states, ssm_state = mamba_inner_fn(
                projected_states,
                self.conv1d.weight,
                self.conv1d.bias if self.use_conv_bias else None,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias.float() if self.use_bias else None,
                -torch.exp(self.A_log.float()),
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                ssm_initial_state = inputs_ssm_states,
                length = None if attention_mask is None else torch.sum(attention_mask, dim=-1, dtype=torch.long)
            )

        else:
            hidden_states, gate = projected_states.chunk(2, dim=1)

            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask.unsqueeze(1)

            # 2. Convolution sequence transformation
            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            if cache_params is not None and cache_position[0] > 0:
                hidden_states = modeling_mamba.causal_conv1d_update(
                    hidden_states.squeeze(-1),
                    cache_params.conv_states[self.layer_idx],
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                )
                hidden_states = hidden_states.unsqueeze(-1)
            else:
                if cache_params is not None:
                    conv_states = nn.functional.pad(
                        hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
                    )
                    cache_params.update_conv_state(self.layer_idx, conv_states, cache_position)
                hidden_states = modeling_mamba.causal_conv1d_fn(
                    hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
                )

            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask.unsqueeze(1)

            # 3. State Space Model sequence transformation
            # 3.a. input varying initialization of time_step, B and C
            ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
            time_step, B, C = torch.split(
                ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
            )
            discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)

            A = -torch.exp(self.A_log.float())
            # 3.c perform the recurrence y ← SSM(A, B, C)(x)
            time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") else None
            if cache_params is not None and cache_position[0] > 0:
                scan_outputs = selective_state_update(
                    cache_params.ssm_states[self.layer_idx],
                    hidden_states[..., 0],
                    discrete_time_step[..., 0],
                    A,
                    B[:, 0],
                    C[:, 0],
                    self.D,
                    gate[..., 0],
                    time_proj_bias,
                    dt_softplus=True,
                ).unsqueeze(-1)
            else:
                scan_outputs, ssm_state = selective_scan_fn(
                    hidden_states,
                    discrete_time_step,
                    A,
                    B.transpose(1, 2),
                    C.transpose(1, 2),
                    self.D.float(),
                    gate,
                    time_proj_bias,
                    delta_softplus=True,
                    return_last_state=True,
                    ssm_initial_state = inputs_ssm_states,
                    length = None if attention_mask is None else torch.sum(attention_mask, dim=-1, dtype=torch.long)
                )
                if ssm_state is not None and cache_params is not None:
                    cache_params.update_ssm_state(self.layer_idx, ssm_state)

            # 4. Final linear projection
            contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))
        return contextualized_states, ssm_state



class MambaBlock(modeling_mamba.MambaBlock):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.mixer = MambaMixer(config, layer_idx = layer_idx)

    def forward(
        self,
        hidden_states,
        inputs_ssm_states = None,
        cache_params = None,
        cache_position = None,
        attention_mask = None,
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states, ssm_last_states = self.mixer(
            hidden_states, 
            inputs_ssm_states=inputs_ssm_states, 
            cache_params=cache_params, 
            cache_position=cache_position, 
            attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states
        return hidden_states, ssm_last_states