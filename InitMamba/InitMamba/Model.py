from . import Block
import importlib
importlib.reload(Block)
from .Block import *
from . import Output
importlib.reload(Output)


class MambaModel(modeling_mamba.MambaModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([MambaBlock(config, layer_idx=idx) 
                                        for idx in range(config.num_hidden_layers)])
        
    @modeling_mamba.add_start_docstrings_to_model_forward(modeling_mamba.MAMBA_INPUTS_DOCSTRING)
    @modeling_mamba.add_code_sample_docstrings(
        checkpoint=modeling_mamba._CHECKPOINT_FOR_DOC,
        output_type=Output.MambaOutput,
        config_class=modeling_mamba._CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: modeling_mamba.Optional[torch.LongTensor] = None,
        inputs_embeds: modeling_mamba.Optional[torch.LongTensor] = None,
        inputs_ssm_states: modeling_mamba.Optional[torch.FloatTensor] = None,
        cache_params: modeling_mamba.Optional[modeling_mamba.MambaCache] = None,
        use_cache: modeling_mamba.Optional[bool] = None,
        output_hidden_states: modeling_mamba.Optional[bool] = None,
        output_ssm_last_states: modeling_mamba.Optional[bool] = None,
        return_dict: modeling_mamba.Optional[bool] = None,
        cache_position: modeling_mamba.Optional[torch.LongTensor] = None,
        attention_mask: modeling_mamba.Optional[torch.LongTensor] = None,
    ) -> modeling_mamba.Union[modeling_mamba.Tuple, Output.MambaOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache:
            if cache_params is None:
                cache_params = modeling_mamba.MambaCache(
                    self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                cache_position = torch.arange(0, self.config.conv_kernel, device=inputs_embeds.device)
            elif cache_position is None:
                # cases when we do manual forward instead of using `model.generate` which will initiate
                # `cache_position` and makes sure it is not None, throw error here instead of doing some
                # hack to conjecture the current cache position
                raise ValueError(
                    "You have to specify the `cache_position` manually when `use_cache=True` and `cache_params` is passed, "
                    "you don't have to pass a `cache_params` if you are in prefilling stage because in that case it will "
                    "be initialized for you automatically"
                )
        else:
            cache_params = None

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_ssm_last_states = () if output_ssm_last_states else None
        for i, mixer_block in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                hidden_states, ssm_last_states = self._gradient_checkpointing_func(
                    mixer_block.__call__, 
                    hidden_states, 
                    inputs_ssm_states[i] if inputs_ssm_states is not None else None, 
                    cache_params, 
                    cache_position, 
                    attention_mask
                )
            else:
                hidden_states, ssm_last_states = mixer_block(
                    hidden_states,
                    inputs_ssm_states=inputs_ssm_states[i] if inputs_ssm_states is not None else None,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if output_ssm_last_states:
                all_ssm_last_states = all_ssm_last_states + (ssm_last_states,)

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states, all_ssm_last_states] if v is not None)

        return Output.MambaOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
            ssm_last_states=all_ssm_last_states,
        )