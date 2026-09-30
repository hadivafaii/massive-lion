"""Regression checks for the numerical and statistical experiment contract."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize as torchvision_normalize, to_tensor
from PIL import Image

from experiments.cifar import parse_args as cifar_args, evaluate_preserving_rng
from experiments.common import language_lr, vision_lr, optimizer_for
from experiments.language_data import concat_chunk, make_dataloader, move_batch_to_device
from experiments.summarize import (best_checkpoint, holm, language_summary,
                                  selected_summary, choose_by_validation)
from experiments.sweep import expand, command
from experiments.vision_data import CIFAR10Loader, MEAN, STD, normalize, split_indices

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    return json.loads((ROOT / 'configs' / f'{name}.json').read_text())


def test_preparation_matches_torchvision_normalization_and_original_split():
    raw = np.arange(3 * 32 * 32, dtype=np.int64).reshape(32, 32, 3).astype(np.uint8)
    expected = torchvision_normalize(to_tensor(Image.fromarray(raw)), MEAN, STD)
    actual = normalize(torch.from_numpy(raw).permute(2, 0, 1).unsqueeze(0))[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    original = torch.utils.data.random_split(range(50000), [45000, 5000],
                                             generator=torch.Generator().manual_seed(0))
    train, valid = split_indices()
    assert train.tolist() == original[0].indices
    assert valid.tolist() == original[1].indices
    assert not set(train.tolist()) & set(valid.tolist())


def test_online_augmentation_matches_raw_black_padding_and_epoch_rng():
    raw = torch.arange(4 * 3 * 32 * 32).reshape(4, 3, 32, 32).remainder(256).to(torch.uint8)
    loader = CIFAR10Loader(normalize(raw), torch.arange(4), 4, train=True, seed=19)
    actual, labels = next(iter(loader))
    generator = torch.Generator().manual_seed(19)
    order = torch.randperm(4, generator=generator)
    offsets = torch.randint(9, (2, 4), generator=generator)
    flips = torch.rand(4, generator=generator) < .5
    padded = F.pad(raw, (4, 4, 4, 4))
    cropped = torch.stack([padded[index, :, top:top+32, left:left+32]
                           for index, top, left in zip(order, offsets[0], offsets[1])])
    cropped[flips] = cropped[flips].flip(-1)
    torch.testing.assert_close(actual, normalize(cropped), rtol=0, atol=1e-6)
    assert torch.equal(labels, order)
    loader.set_epoch(0)
    replay, _ = next(iter(loader))
    assert torch.equal(actual, replay)


def test_schedules_preserve_distinct_first_update_conventions():
    assert vision_lr(0, .008, 1e-5, 80000) == .008
    assert vision_lr(40000, .008, 1e-5, 80000) == pytest.approx(.004005)
    assert language_lr(0, .016, 1e-5, 620, 6200) == 0
    assert language_lr(620, .016, 1e-5, 620, 6200) == .016
    assert language_lr(6200, .016, 1e-5, 620, 6200) == 1e-5
    assert language_lr(6199, .016, 1e-5, 620, 6200) > 1e-5


@pytest.mark.parametrize('name,count', [('cifar_selected',70),('cifar_adaptive_mass',576),
    ('cifar_fixed_mass',1248),('cifar_kinematics',2736),('cifar_cutoffs',16),('language_confirmation',112)])
def test_grids_are_unique_and_complete(name, count):
    jobs = expand(config(name))
    assert len(jobs) == count
    assert len({json.dumps(job,sort_keys=True) for job in jobs}) == count
    if name == 'language_confirmation':
        assert {job['seed'] for job in jobs} == set(range(100,107))
        assert sum(job['grad_clip'] == 0 for job in jobs) == 21
        assert {job['mass'] for job in jobs} == {0}
    if name == 'cifar_kinematics':
        assert {job['kinematics'] for job in jobs} == {'minkowski','arctan','tanh'}


def test_named_reductions_log_actual_settings_and_reject_conflicts(tmp_path):
    args = cifar_args(['--output', str(tmp_path), '--optim', 'lion'])
    assert args.mass == 0 and not args.adaptive_mass
    assert args.label == 'Lion'
    args = cifar_args(['--output', str(tmp_path), '--optim', 'ss_adamw'])
    assert args.beta1 == args.beta2 and args.mass == 0 and args.adaptive_mass
    assert args.label == 'SS AdamW' and args.section == 'Signum'
    param = torch.nn.Parameter(torch.ones(1))
    optimizer = optimizer_for([param], args)
    assert optimizer.param_groups[0]['mass'] == args.mass
    with pytest.raises(ValueError, match='mass=0'):
        cifar_args(['--output', str(tmp_path), '--optim', 'lion', '--mass', '1'])


def test_every_selected_job_parses_and_constructs(tmp_path):
    for job in expand(config('cifar_selected'))[:10]:
        cli = command('cifar', job, tmp_path)[3:]
        args = cifar_args(cli)
        optimizer_for([torch.nn.Parameter(torch.ones(1))], args)


def test_validation_selection_uses_earliest_checkpoint_and_ignores_test():
    history = [dict(step=10, validation_correct=9,validation_accuracy=90,test_accuracy=70),
               dict(step=20, validation_correct=9,validation_accuracy=90,test_accuracy=99),
               dict(step=30, validation_correct=8,validation_accuracy=80,test_accuracy=100)]
    assert best_checkpoint(history)['step'] == 10
    assert best_checkpoint(history, cutoff=20)['test_accuracy'] == 70


def test_hyperparameter_selection_does_not_use_test():
    frame = pd.DataFrame([
        dict(mass=1,beta2=.9,selected_step=20,run_index=0,validation_accuracy=95,test_accuracy=80),
        dict(mass=1,beta2=.9,selected_step=20,run_index=1,validation_accuracy=94,test_accuracy=99)])
    assert choose_by_validation(frame,['mass','beta2']).iloc[0].run_index == 0


def test_lm_chunking_is_per_map_batch_and_targets_are_shifted():
    rows = concat_chunk({'input_ids': [[1,2,3], [4,5,6,7]]}, max_seq_length=4)
    assert rows['input_ids'] == [[1,2,3,4]]  # incomplete tail is intentionally discarded
    assert rows['docs_lengths'] == [[3,1]]
    batch = {'input_ids': torch.tensor([[1,2,3,4]])}
    inputs, targets, mask = move_batch_to_device(batch, 3, 'cpu')
    assert inputs.tolist() == [[1,2,3]] and targets.tolist() == [[2,3,4]]
    assert mask is None


def test_language_validation_drops_incomplete_microbatch():
    rows = [{'input_ids': torch.tensor([i,i+1,i+2])} for i in range(5)]
    loader = make_dataloader(rows, micro_batch_size=2, num_workers=0, drop_last=True)
    assert [batch['input_ids'][:,0].tolist() for batch in loader] == [[0,1],[2,3]]
    assert 100000000 // 2049 // 32 * 32 * 2048 == 99942400


def test_holm_known_example():
    np.testing.assert_allclose(holm([.01,.04,.03]), [.03,.06,.06])


def test_language_inference_excludes_selection_seed_and_averages_perplexities():
    rows=[]
    for job in expand(config('language_confirmation')):
        zeta=1-job['beta1']/job['beta2']
        loss=3.0+.001*(job['seed']-100)-zeta*.05*(1+.02*(job['seed']-100))
        if job['seed']==100 and zeta>0:
            loss+=2  # cannot influence paired fresh-seed inference
        rows.append(dict(job,validation_loss=loss,validation_perplexity=np.exp(loss)))
    frame=pd.DataFrame(rows)
    summary,comparisons=language_summary(frame)
    assert len(summary)==16 and len(comparisons)==10
    assert (comparisons.n_seeds==6).all()
    assert (summary.fresh_n_seeds==6).all()
    assert (comparisons.mean_difference<0).all()
    target=frame[(frame.beta1==.938125)&(frame.beta2==.95)&(frame.grad_clip==1)]
    output=summary[(summary.beta1==.938125)&(summary.beta2==.95)&(summary.grad_clip==1)].iloc[0]
    assert output.mean_validation_perplexity==pytest.approx(target.validation_perplexity.mean())
    assert output.mean_validation_perplexity!=pytest.approx(np.exp(target.validation_loss.mean()))


def test_cifar_inference_has_13_planned_fresh_seed_contrasts():
    rows=[]
    labels=[]
    for job in expand(config('cifar_selected')):
        if job['label'] not in labels:
            labels.append(job['label'])
        score=90+labels.index(job['label'])*(.1+job['seed']*.001)+job['seed']*.01
        rows.append(dict(job,validation_accuracy=score,validation_loss=.2,
                         test_accuracy=score,test_loss=.3))
    summary,comparisons=selected_summary(pd.DataFrame(rows))
    assert len(summary)==10 and len(comparisons)==13
    assert (summary.n_seeds==7).all() and (comparisons.n_seeds==6).all()


def test_test_evaluation_preserves_rng_and_original_accuracy_arithmetic():
    import contextlib
    import random

    class RandomScores(torch.nn.Module):
        def forward(self, images):
            random.random()
            np.random.random()
            return torch.rand(len(images), 10)

    model = RandomScores()
    torch.manual_seed(9)
    random.seed(9)
    np.random.seed(9)
    torch_before = torch.get_rng_state()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    result = evaluate_preserving_rng(model, [(torch.ones(3, 1), torch.zeros(3,dtype=torch.long))],
                                     contextlib.nullcontext(), torch.device('cpu'))
    assert torch.equal(torch_before, torch.get_rng_state())
    assert python_before == random.getstate()
    assert np.array_equal(numpy_before[1], np.random.get_state()[1])
    assert result['accuracy'] == result['correct'] / result['count'] * 100
