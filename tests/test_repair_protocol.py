"""Regression checks for fixed sampling and the original collapse failure mode."""
import unittest
import numpy as np
import torch
from colab.repair_runner import fixed_indices, collapse_signal, health_check
from us_pipeline.models import CNN78u


class RepairProtocolTests(unittest.TestCase):
    def test_fixed_uniform_subset_is_reusable_and_not_stride_five(self):
        a=fixed_indices(1720827)
        self.assertEqual(len(a),344165)
        np.testing.assert_array_equal(a,fixed_indices(1720827))
        self.assertEqual(len(np.unique(a)),len(a))
        self.assertTrue(np.all(np.diff(a)>0))
        self.assertGreater(len(np.unique(a%5)),1)
        self.assertFalse(np.array_equal(a,fixed_indices(1720827,seed=70)))

    def test_constant_predictions_alone_do_not_trigger_collapse_guard(self):
        h={'train':{'probability_std':0.},'validation':{'probability_std':0.},
           'feature_parameter_max_abs':.1}
        self.assertFalse(collapse_signal(h))
        h['feature_parameter_max_abs']=1e-40
        self.assertTrue(collapse_signal(h))

    def test_monitor_detects_zero_features_despite_nonzero_bias_and_preserves_state(self):
        torch.set_num_threads(2)
        model=CNN78u(channels=(2,4))
        with torch.no_grad():
            for name,p in model.blocks.named_parameters():
                p.fill_(0 if name.endswith('weight') else .1)
        model.train()
        state={k:v.clone() for k,v in model.state_dict().items()}
        probes={name:(torch.rand(2,1,234,234),torch.tensor([0,1]))
                for name in ('train','validation')}
        rng=torch.get_rng_state().clone()
        h=health_check(model,probes,torch.device('cpu'),{},1)
        self.assertTrue(h['collapse_signal'])
        self.assertTrue(model.training)
        torch.testing.assert_close(rng,torch.get_rng_state())
        for name,v in model.state_dict().items():
            torch.testing.assert_close(v,state[name])


if __name__=='__main__':
    unittest.main()
