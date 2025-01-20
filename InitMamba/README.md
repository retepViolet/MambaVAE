# InitMamba

We created a subclass for the Mamba in the huggingface transformers, which is capable of setting the initial states of each SSM layers and returning the last states of each SSM layers.

## Parameters

- inputs_ssm_states: a tuple of the initial SSM states of each layer.

- output_hidden_states: a boolean value indicating whether to return the last SSM states of each layer.

- others: same as the Mamba in the huggingface transformers.

## Output

- ssm_last_states: a tuple of the last SSM states of each layer.

- others: same as the Mamba in the huggingface transformers.

## Usage

See examples in the "lab.ipynb".