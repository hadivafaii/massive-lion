"""The exact name-based weight-decay rule for the language model."""

def parameter_groups(model, weight_decay):
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    named_params = {
        name: param for name, param in model.named_parameters()
        if param.requires_grad
    }
    decay_names = [
        name for name, param in named_params.items()
        if (
            not getattr(param, "_no_weight_decay", False)
            and "bias" not in name
            and "norm" not in name
        )
    ]
    decay_name_set = set(decay_names)
    decay_params = [
        param for name, param in named_params.items()
        if name in decay_name_set
    ]
    no_decay_params = [
        param for name, param in named_params.items()
        if name not in decay_name_set
    ]
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
