import unittest
import numpy as np
import pandas as pd
from us_pipeline.data import sample_indices,read_config
from us_pipeline.images import render_window,prediction_basis


def config(**kwargs):
    c=dict(lookback=3,horizon=2,stride=1,allow_cross_session=True,zero_return_epsilon=0.,
           train_end='2023-01-03',validation_end='2023-01-04')
    c.update(kwargs);return c


def clock():
    return pd.DatetimeIndex([t for day in ['2023-01-03','2023-01-04','2023-01-05']
        for t in pd.date_range(day+' 09:35',periods=6,freq='5min',tz='America/New_York')])


class WindowTests(unittest.TestCase):
    def test_chronology_and_purging_use_label_endpoint(self):
        times=clock();valid=np.ones((1,len(times)),bool)
        close=np.arange(1,len(times)+1,dtype=float)[None,:]
        splits,stats=sample_indices(valid,close,times,config(),[99])
        self.assertEqual(stats['purged_split_boundary'],4)
        self.assertEqual(splits['train'].end_index.tolist(),[2,3])
        self.assertEqual(splits['validation'].end_index.tolist(),[6,7,8,9])
        self.assertTrue(splits['train'].label_end_time.max()<splits['validation'].decision_time.min())
        self.assertTrue((splits['train'].label==1).all())
        # H=2 means exactly two bars, never H+1.
        self.assertAlmostEqual(splits['train'].forward_return.iloc[0],5/3-1)

    def test_missing_bar_not_removed_or_bridged_in_inputs_or_targets(self):
        times=clock();valid=np.ones((1,len(times)),bool);valid[0,3]=False
        close=np.arange(1,len(times)+1,dtype=float)[None,:]
        splits,stats=sample_indices(valid,close,times,config(),[99])
        for split in splits.values():
            for t in split.end_index:
                self.assertTrue(valid[0,t-2:t+3].all())
        self.assertGreater(stats['invalid_input'],0)
        self.assertGreater(stats['invalid_future'],0)

    def test_neutral_labels_are_excluded_without_balancing(self):
        times=clock();valid=np.ones((1,len(times)),bool)
        splits,stats=sample_indices(valid,np.ones_like(valid,dtype=float),times,config(),[99])
        self.assertEqual(stats['kept'],0)
        self.assertGreater(stats['zero_return'],0)

    def test_labels_follow_return_sign(self):
        times=clock();valid=np.ones((1,len(times)),bool)
        close=np.arange(len(times),0,-1,dtype=float)[None,:]
        splits,_=sample_indices(valid,close,times,config(),[99])
        self.assertTrue((splits['train'].label==0).all())

    def test_next_bar_label_is_t_plus_one_not_two(self):
        times=clock();valid=np.ones((1,len(times)),bool)
        close=np.arange(100,100+len(times),dtype=float)[None,:]
        close[0,2:5]=[100.,110.,90.]
        splits,stats=sample_indices(valid,close,times,config(horizon=1),[99])
        row=splits['train'].set_index('end_index').loc[2]
        self.assertEqual(row.label,1)
        self.assertAlmostEqual(row.forward_return,.1)
        self.assertEqual(row.label_end_time,times[3])
        self.assertEqual(stats['purged_split_boundary'],2)

    def test_next_missing_bar_is_never_skipped_for_a_later_bar(self):
        times=clock();valid=np.ones((1,len(times)),bool);valid[0,3]=False
        close=np.arange(100,100+len(times),dtype=float)[None,:]
        splits,stats=sample_indices(valid,close,times,config(horizon=1),[99])
        for frame in splits.values():
            self.assertFalse(frame.end_index.eq(2).any())
        self.assertGreater(stats['invalid_future'],0)

    def test_cross_session_can_be_disabled(self):
        times=clock();valid=np.ones((1,len(times)),bool)
        close=np.arange(1,len(times)+1,dtype=float)[None,:]
        splits,stats=sample_indices(valid,close,times,config(allow_cross_session=False),[99])
        self.assertGreater(stats['cross_session_disallowed'],0)
        for frame in splits.values():
            for t in frame.end_index:
                self.assertEqual(times[t-2].date(),times[t+2].date())

    def test_active_config_uses_full_session_and_rejects_legacy_windows(self):
        import json,tempfile
        from pathlib import Path
        active=read_config(Path(__file__).resolve().parents[1]/'configs/us_5min.json')
        self.assertEqual(active['lookback'],390//5)
        for changes in [{'lookback':49},{'lookback':9},{'bar_minutes':30}]:
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'config.json'
                path.write_text(json.dumps({**active,**changes}))
                with self.assertRaises(ValueError):
                    read_config(path)


class ImageTests(unittest.TestCase):
    def test_shared_price_axis_and_high_low_geometry(self):
        x=np.array([[10.,12.,8.,11.,100.],[11.,13.,9.,12.,200.],[12.,14.,10.,13.,150.]])
        image=render_window(x,False)
        self.assertEqual(image.shape,(9,9))
        # The shared maximum (third high) is the top, the shared minimum bottom.
        self.assertEqual(image[0,7],255)
        self.assertEqual(image[8,1],255)

    def test_window_normalization_is_scale_invariant_and_causal(self):
        x=np.array([[10.,12.,8.,11.,100.],[11.,13.,9.,12.,200.],[12.,14.,10.,13.,150.]])
        future_scaled=x.copy();future_scaled[:,:4]*=.5;future_scaled[:,4]*=2
        recovered=prediction_basis(future_scaled,[.5,2.])
        np.testing.assert_allclose(recovered,x)
        np.testing.assert_array_equal(render_window(recovered),render_window(x))

    def test_flat_prices_render_without_division_by_zero(self):
        image=render_window(np.tile([10.,10.,10.,10.,100.],(9,1)))
        self.assertEqual(image.shape,(27,27));self.assertGreater(image.sum(),0)

    def test_missing_data_never_enters_renderer(self):
        x=np.tile([10.,12.,8.,11.,100.],(9,1));x[2,3]=np.nan
        with self.assertRaises(ValueError):
            render_window(x)


class ModelTests(unittest.TestCase):
    def test_cnn78u_shape_and_finite_gradients(self):
        import torch
        from us_pipeline.models import CNN78u
        torch.set_num_threads(2)
        model=CNN78u();out=model(torch.rand(2,1,234,234))
        self.assertEqual(tuple(out.shape),(2,2))
        self.assertEqual(len(model.channels),4)
        self.assertEqual(model.convolutional_receptive_field,(76,46))
        loss=torch.nn.functional.cross_entropy(out,torch.tensor([0,1]));loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        with self.assertRaises(ValueError):
            model(torch.rand(2,1,147,147))

    def test_same_model_supports_controlled_depth_configuration(self):
        import torch
        from us_pipeline.models import CNN78u
        shallow=CNN78u(channels=(32,64))
        self.assertEqual(shallow.convolutional_receptive_field,(16,10))
        self.assertEqual(tuple(shallow(torch.rand(2,1,234,234)).shape),(2,2))


if __name__=='__main__':
    unittest.main()
