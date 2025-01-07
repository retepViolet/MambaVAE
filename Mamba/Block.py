import torch
from torch import Tensor, nn
from torch.nn import functional as F
import transformers
from transformers.models.mamba import modeling_mamba


class MambaMixer(modeling_mamba.MambaMixer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
    
    def forward(self, input_states, inputs_ssm_states=None, output_ssm_last_states=None,
                cache_params=None, cache_position=None, attention_mask=None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(input_states).transpose(1, 2)                   # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, dim=1)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 2. Convolution sequence transformation
        if inputs_ssm_states is not None: 
            # Most important revise !!!!!!!!!!!!!!!
            ssm_state = inputs_ssm_states
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
        elif cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            ssm_state = ssm_state.to(hidden_states.device)
            # use `cache_position.shape[0]` to check whether we are in prefill
            # stage, it's equivalent to check `cache_position[0] == 0`, which
            # breaks dynamo fullgraph constraints
            if cache_position.shape[0] == self.conv_kernel_size:
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )

                cache_params.update_conv_state(self.layer_idx, conv_state, cache_position)
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])     # [batch, intermediate_size, seq_len]
            else:
                conv_state = cache_params.update_conv_state(self.layer_idx, hidden_states, cache_position)
                hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)         # [batch, intermediate_size, 1] : decoding
        else:
            ssm_state = torch.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size),
                device=hidden_states.device, dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])         # [batch, intermediate_size, seq_len]
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        discrete_time_step = self.dt_proj(time_step)                                    # [batch, seq_len, intermediate_size]
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2) # [batch, intermediate_size, seq_len]

        # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        A = -torch.exp(self.A_log.float())                                              # [intermediate_size, ssm_state_size]
        discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()       # [batch, intermediate_size, seq_len, ssm_state_size]
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()
        
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        hs = []
        if self.training and cache_params is None:
            # Scan
            # Most important revise !!!!!!!!!!!!!!!
            discrete_A, deltaB_u = discrete_A.transpose(1, 2), deltaB_u.transpose(1, 2)
            if inputs_ssm_states is not None: # [batch, intermediate_size, ssm_state_size]
                A0X0 = discrete_A[:, 0, :, :] * inputs_ssm_states
                deltaB_u[:, 0, :, :] = deltaB_u[:, 0, :, :] + A0X0 # Add initial state to the first X before scan
            
            hs = modeling_mamba.pscan(discrete_A, deltaB_u) # [batch, seq_len, intermediate_size, ssm_state_size]
            scan_output = (hs @ C.unsqueeze(-1)).squeeze(3).transpose(1, 2) # [batch, intermediate_size, seq_len]
            scan_output = scan_output + hidden_states * self.D[None, :, None]
            scan_output = scan_output * self.act(gate)
        else: 
            # Loop
            print('using loop')
            scan_outputs = []
            for i in range(seq_len):
                ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediade_size, ssm_state]
                if output_ssm_last_states: hs.append(ssm_state)
                scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
                scan_outputs.append(scan_output[:, :, 0])
            
            if output_ssm_last_states: hs = torch.stack(hs, dim=1)
            scan_output = torch.stack(scan_outputs, dim=-1)                                # [batch, seq_len, intermediade_size]
            scan_output = scan_output + (hidden_states * self.D[None, :, None])
            scan_output = (scan_output * self.act(gate))

            if cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        ssm_last_states = None
        if output_ssm_last_states:
            if attention_mask is None: ssm_last_states = hs[:, -1]
            else: ssm_last_states = hs[torch.arange(batch_size), torch.sum(attention_mask, dim=-1, dtype=torch.int64) - 1]

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.transpose(1, 2))  # [batch, seq_len, hidden_size]
        return contextualized_states, ssm_last_states



class MambaBlock(modeling_mamba.MambaBlock):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.mixer = MambaMixer(config, layer_idx = layer_idx)

    def forward(
        self,
        hidden_states,
        inputs_ssm_states = None,
        output_ssm_last_states = None,
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
            output_ssm_last_states = output_ssm_last_states,
            cache_params=cache_params, 
            cache_position=cache_position, 
            attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states
        return hidden_states, ssm_last_states