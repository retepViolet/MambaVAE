from transformers.models.mamba import modeling_mamba


@modeling_mamba.dataclass
class MambaCausalLMOutput(modeling_mamba.ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (`MambaCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: modeling_mamba.Optional[modeling_mamba.torch.FloatTensor] = None
    logits: modeling_mamba.Optional[modeling_mamba.torch.FloatTensor] = None
    cache_params: modeling_mamba.Optional[modeling_mamba.MambaCache] = None
    hidden_states: modeling_mamba.Optional[modeling_mamba.Tuple[modeling_mamba.torch.FloatTensor]] = None
    ssm_last_states: modeling_mamba.Optional[
        modeling_mamba.Union[
            modeling_mamba.Tuple[modeling_mamba.torch.FloatTensor],
            modeling_mamba.torch.FloatTensor,
        ]
    ] = None


@modeling_mamba.dataclass
class MambaOutput(modeling_mamba.ModelOutput):
    """
    Class for the MAMBA model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`MambaCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: modeling_mamba.Optional[modeling_mamba.torch.FloatTensor] = None
    cache_params: modeling_mamba.Optional[modeling_mamba.MambaCache] = None
    hidden_states: modeling_mamba.Optional[modeling_mamba.Tuple[modeling_mamba.torch.FloatTensor]] = None
    ssm_last_states: modeling_mamba.Optional[
        modeling_mamba.Union[
            modeling_mamba.Tuple[modeling_mamba.torch.FloatTensor],
            modeling_mamba.torch.FloatTensor,
        ]
    ] = None