from . import Model
import importlib
importlib.reload(Model)
from .Model import *


class MambaForCausalLM(modeling_mamba.MambaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = MambaModel(config)

    @modeling_mamba.add_start_docstrings_to_model_forward(modeling_mamba.MAMBA_INPUTS_DOCSTRING)
    @modeling_mamba.add_code_sample_docstrings(
        checkpoint=modeling_mamba._CHECKPOINT_FOR_DOC,
        output_type=Output.MambaCausalLMOutput,
        config_class=modeling_mamba._CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: modeling_mamba.Optional[torch.LongTensor] = None,
        attention_mask: modeling_mamba.Optional[torch.LongTensor] = None,
        inputs_embeds: modeling_mamba.Optional[torch.FloatTensor] = None,
        layer_range: modeling_mamba.Optional[range] = None,
        inputs_ssm_states: modeling_mamba.Optional[torch.FloatTensor] = None,
        cache_params: modeling_mamba.Optional[modeling_mamba.MambaCache] = None,
        labels: modeling_mamba.Optional[torch.LongTensor] = None,
        output_hidden_states: modeling_mamba.Optional[bool] = None,
        output_ssm_last_states: modeling_mamba.Optional[bool] = None,
        return_dict: modeling_mamba.Optional[bool] = None,
        use_cache: modeling_mamba.Optional[bool] = None,
        cache_position: modeling_mamba.Optional[torch.Tensor] = None,
        **kwargs,  # for now we need this for generation
    ) -> modeling_mamba.Union[modeling_mamba.Tuple, Output.MambaCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mamba_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            layer_range=layer_range,
            inputs_ssm_states=inputs_ssm_states,
            output_hidden_states=output_hidden_states,
            output_ssm_last_states=output_ssm_last_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = mamba_outputs[0]

        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = modeling_mamba.CrossEntropyLoss(reduction = 'none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            if attention_mask is not None:
                shift_mask = attention_mask[..., :-1, :].contiguous()
                loss = (loss * shift_mask).sum() / attention_mask.sum()
            else:
                loss = loss.mean()

        if not return_dict:
            output = (logits,) + mamba_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Output.MambaCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
            ssm_last_states=mamba_outputs.ssm_last_states,
        )