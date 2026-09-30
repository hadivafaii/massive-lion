"""Weight-decay grouping from the ResNet training recipe."""
import torch


def parameter_groups(model, weight_decay):
    decay = set()
    no_decay = set()

    # Track seen parameter tensors to handle weight tying
    seen_param_ids = set()

    # Modules whose weights SHOULD be decayed
    whitelist_weight_modules = (
        torch.nn.Linear,
        torch.nn.Conv2d,
    )

    # Modules whose weights should NOT be decayed
    blacklist_weight_modules = (
        torch.nn.Embedding,  # controversial
        torch.nn.LayerNorm,
        torch.nn.BatchNorm2d,
        torch.nn.GroupNorm,
        torch.nn.RMSNorm,
    )

    # Build mapping from param name to param
    # (only first occurrence for tied weights)
    param_dict = {}

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            if not p.requires_grad:
                continue

            # Skip shared parameters (weight tying)
            # Use id() to detect same tensor with different names
            if id(p) in seen_param_ids:
                continue
            seen_param_ids.add(id(p))

            fpn = f'{mn}.{pn}' if mn else pn
            param_dict[fpn] = p

            if pn.endswith('bias'):
                # All biases: no decay
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # Linear, Conv2d weights: decay
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # LayerNorm, BatchNorm, Embedding weights: no decay
                no_decay.add(fpn)
            else:
                # Fallback for custom params (pos_emb, gamma, logit_scale, etc.)
                if p.dim() < 2:
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)

    # Validation
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, \
        f"Parameters {inter_params} in both decay/no_decay!"
    assert len(param_dict.keys() - union_params) == 0, \
        f"Parameters {param_dict.keys() - union_params} not categorized!"

    # Create optimizer groups
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(decay)],
         "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(no_decay)],
         "weight_decay": 0.0},
    ]

    return optim_groups
