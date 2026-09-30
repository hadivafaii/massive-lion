import json
import unittest
from dataclasses import asdict

import torch

from dynamics_lab.engine import SimulationState, OptimizerSpec, build_optimizer, defaults_payload

ADAPTIVE_NAME = 'MassiveLion'
LEGACY_MASS_MODE = 'momentum_diff'

class OptimizerInstanceTests(unittest.TestCase):
    def config(self, mode='parallel'):
        return {
            'mode': mode, 'max_steps': 3, 'theta0': [-1.8, 2.3],
            'ensemble_count': 3, 'ensemble_optimizer': 'MassiveLion',
            'noise': {'enabled': False},
            'optimizers': [
                {'name': 'MassiveLion', 'instance_id': 'slow', 'lr': .01,
                 'mass': .2, 'label': 'My slow run', 'color': '#12aBcD'},
                {'name': 'MassiveLion', 'instance_id': 'fast', 'lr': .05,
                 'mass': .2, 'display_label': 'MassiveLion · lr=0.05', 'color': '#F80'},
            ],
        }

    def test_duplicate_algorithms_remain_independent_and_roundtrip(self):
        state = SimulationState()
        initial = state.reset(self.config())
        self.assertEqual([row['id'] for row in initial['learners']], ['slow', 'fast'])
        self.assertEqual([row['instance_id'] for row in initial['learners']], ['slow', 'fast'])
        self.assertEqual([row['color'] for row in initial['learners']], ['#12abcd', '#ff8800'])
        self.assertIsNone(initial['learners'][1]['label'])
        self.assertEqual(initial['learners'][1]['name'], 'MassiveLion · lr=0.05')
        result = state.step()
        self.assertNotEqual(result['learners'][0]['trace'][-1]['theta'],
                            result['learners'][1]['trace'][-1]['theta'])
        self.assertIsNot(state.learners[0].optimizer, state.learners[1].optimizer)
        self.assertIsNot(state.learners[0].theta, state.learners[1].theta)
        config = self.config()
        config['optimizers'] = json.loads(json.dumps(initial['optimizers']))
        restored = SimulationState().reset(config)
        self.assertEqual(restored['optimizers'], initial['optimizers'])
        self.assertEqual([row['name'] for row in restored['learners']],
                         [row['name'] for row in initial['learners']])
        config['optimizers'].reverse()
        reordered = SimulationState().reset(config)
        self.assertEqual([row['id'] for row in reordered['learners']], ['fast', 'slow'])
        self.assertEqual(reordered['learners'][0]['color'], '#ff8800')

    def test_serial_duplicate_instances_have_unique_stable_ids(self):
        state = SimulationState()
        snapshot = state.reset(self.config('serial'))
        while not snapshot['done']:
            snapshot = state.step()
        self.assertEqual([row['id'] for row in snapshot['learners']], ['slow', 'fast'])
        self.assertEqual([row['local_step'] for row in snapshot['learners']], [3, 3])

    def test_ensemble_targets_selected_instance_and_distinguishes_members(self):
        config = self.config('ensemble')
        config['ensemble_instance_id'] = 'fast'
        state = SimulationState()
        snapshot = state.reset(config)
        self.assertEqual(snapshot['ensemble_instance_id'], 'fast')
        self.assertEqual([row['id'] for row in snapshot['learners']],
                         ['fast:member-1', 'fast:member-2', 'fast:member-3'])
        for row in snapshot['learners']:
            self.assertEqual(row['instance_id'], 'fast')
            self.assertEqual(row['lr'], .05)
            self.assertEqual(row['color'], '#ff8800')
        self.assertEqual(len({id(learner.optimizer) for learner in state.learners}), 3)
        legacy = SimulationState().reset(self.config('ensemble'))
        self.assertEqual(legacy['ensemble_instance_id'], 'slow')
        config['ensemble_instance_id'] = 'missing'
        with self.assertRaisesRegex(ValueError, 'ensemble_instance_id'):
            SimulationState().reset(config)

    def test_legacy_rows_receive_collision_free_ids(self):
        config = self.config()
        config['optimizers'] = [
            {'name': 'MassiveLion'}, {'name': 'MassiveLion', 'instance_id': 'optimizer-1'},
            {'name': 'MassiveLion'},
        ]
        result = SimulationState().reset(config)
        identities = [row['instance_id'] for row in result['optimizers']]
        self.assertEqual(len(set(identities)), 3)
        config['optimizers'] = result['optimizers']
        self.assertEqual(SimulationState().reset(config)['optimizers'], result['optimizers'])
        config['optimizers'][0]['instance_id'] = 'optimizer-1'
        with self.assertRaisesRegex(ValueError, 'Duplicate optimizer instance_id'):
            SimulationState().reset(config)

    def test_display_text_does_not_redefine_optimizer_family(self):
        spec = OptimizerSpec.from_dict({'name': 'MassiveLion', 'mass': .2,
                                      'label': 'Lion', 'display_label': 'Signum'})
        self.assertEqual(spec.display_name(), 'Lion')
        self.assertEqual(spec.display_family(), 'MassiveLion')
        state = SimulationState()
        snapshot = state.reset({'mode': 'ensemble', 'ensemble_optimizer': 'Lion',
                                'ensemble_count': 1, 'max_steps': 1,
                                'optimizers': [asdict(spec)]})
        self.assertEqual(snapshot['learners'][0]['mass'], 0.)
        self.assertIsNone(OptimizerSpec.from_dict({'name': 'Lion'}).label)
        generated = OptimizerSpec.from_dict({'name': 'MassiveLion', 'display_label': 'A generated title'})
        self.assertEqual(generated.display_name(), 'A generated title')
        self.assertIsNone(generated.label)

    def test_presentation_fields_are_validated(self):
        invalid = [
            {'color': 'red; background:url(https://example.com)'}, {'color': 'red'},
            {'color': '#12345'}, {'color': '" onload="x'},
            {'instance_id': 'bad id'}, {'instance_id': ['row']},
            {'label': {'text': 'name'}}, {'display_label': ['name']}, {'label': 'two\nlines'},
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                OptimizerSpec.from_dict({'name': 'MassiveLion', **values})
        row = OptimizerSpec.from_dict({'name': 'MassiveLion', 'label': '  ', 'color': '#aBc'})
        self.assertIsNone(row.label)
        self.assertEqual(row.color, '#aabbcc')

    def test_mass_controls_tie_and_untie_independently(self):
        base = {'name': ADAPTIVE_NAME, 'beta1': .5, 'beta2': .81, 'beta3': .6,
                'mass': .2, 'mass_mode': 'momentum_diff', 'adaptive_mass': True,
                'kappa': .2, 'beta_gravity': .7}
        for tied, coupling, decay in [(True, .81, .81), (False, .2, .7)]:
            with self.subTest(tied=tied):
                spec = OptimizerSpec.from_dict({**base, 'tie_mass': tied})
                self.assertEqual(spec.kappa, coupling)
                self.assertEqual(spec.beta_gravity, decay)
                self.assertEqual(spec.beta3, decay)
                self.assertEqual(OptimizerSpec.from_dict(asdict(spec)), spec)
                parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
                group = build_optimizer(spec, parameter).param_groups[0]
                self.assertEqual(group['kappa'], coupling)
                self.assertEqual(group['beta_gravity'], decay)
        disabled = OptimizerSpec.from_dict({**base, 'adaptive_mass': False, 'tie_mass': True})
        self.assertEqual((disabled.kappa, disabled.beta_gravity, disabled.beta3), (0., 0., 0.))

    def test_old_mass_controls_keep_independent_values(self):
        spec = OptimizerSpec.from_dict({'name': ADAPTIVE_NAME, 'beta2': .81,
                                      'beta3': .6, 'kappa': .2})
        self.assertFalse(spec.tie_mass)
        self.assertEqual((spec.kappa, spec.beta_gravity), (.2, .6))
        self.assertEqual(spec.mass_mode, LEGACY_MASS_MODE)

    def test_new_adaptive_ui_defaults_use_paper_ties(self):
        defaults = defaults_payload()['optimizers'][ADAPTIVE_NAME]
        spec = OptimizerSpec.from_dict({'name': ADAPTIVE_NAME, **defaults})
        self.assertTrue(spec.adaptive_mass)
        self.assertTrue(spec.tie_mass)
        self.assertEqual(spec.mass_mode, 'momentum_diff')
        self.assertEqual((spec.kappa, spec.beta_gravity), (spec.beta2, spec.beta2))
