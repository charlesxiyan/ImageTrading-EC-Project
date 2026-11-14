import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from us_pipeline.reporting import binary_metrics,calibration_table,paired_block_interval


class MetricTests(unittest.TestCase):
    def test_known_constant_probability_and_calibration(self):
        y=np.array([0,1,0,1]);p=np.full(4,.5)
        m=binary_metrics(y,p)
        self.assertEqual(m['accuracy'],.5)
        self.assertEqual(m['balanced_accuracy'],.5)
        self.assertEqual(m['roc_auc'],.5)
        self.assertEqual(m['average_precision_up'],.5)
        self.assertAlmostEqual(m['log_loss'],np.log(2))
        self.assertEqual(m['brier_score'],.25)
        table,ece=calibration_table(y,p)
        self.assertEqual(table.samples.sum(),4);self.assertEqual(ece,0.)

    def test_perfect_and_invalid_probabilities(self):
        m=binary_metrics([0,1,0,1],[0,1,0,1])
        self.assertEqual(m['accuracy'],1.);self.assertEqual(m['brier_score'],0.)
        for p in [[float('nan'),.5],[-.1,.5],[1.1,.5]]:
            with self.assertRaises(ValueError):
                binary_metrics([0,1],p)

    def test_block_interval_keeps_paired_daily_outcomes(self):
        dates=pd.date_range('2023-05-01',periods=8,freq='D',tz='America/New_York')
        frame=pd.DataFrame({'decision_time':dates.repeat(2),'label':[1]*16,'probability_up':[.9]*16})
        r=paired_block_interval(frame,0,block_days=3,repetitions=50,seed=3)
        self.assertEqual(r['estimate_accuracy_difference'],1.)
        self.assertEqual(r['lower_95'],1.);self.assertEqual(r['upper_95'],1.)
        self.assertEqual(r['decision_days'],8)


def synthetic_prepared(root):
    """Small fictional data; no actual held-out market data are used in tests."""
    from us_pipeline.models import CNN78u
    t=312;close=100+3*np.sin(np.arange(t)*.7)
    x=np.zeros((1,t,5));x[0,:,0]=close-.1;x[0,:,1]=close+.5
    x[0,:,2]=close-.5;x[0,:,3]=close;x[0,:,4]=100+np.arange(t)
    np.save(root/'features.npy',x);np.save(root/'valid.npy',np.ones((1,t),bool))
    np.save(root/'multipliers.npy',np.ones((1,t,2)))
    clock=pd.DatetimeIndex([stamp for day in ['2023-01-03','2023-01-04','2023-01-05','2023-01-06']
        for stamp in pd.date_range(day+' 09:35',periods=78,freq='5min',tz='America/New_York')])
    config={'lookback':78,'horizon':1,'bar_minutes':5,'include_volume':True,
        'model':{'name':'CNN78u','channels':[2,4,8,16],'pool_grid':2,'dropout':.5}}
    for split,ends in [('train',np.arange(77,89)),('validation',np.arange(156,164)),('test',np.arange(234,242))]:
        r=close[ends+1]/close[ends]-1
        frame=pd.DataFrame({'security_index':0,'permno':99,'end_index':ends,
            'label':(r>0).astype('int8'),'forward_return':r,
            'decision_time':clock[ends],'label_end_time':clock[ends+1]})
        frame.to_parquet(root/f'{split}.parquet',index=False)
    (root/'manifest.json').write_text(json.dumps({'status':'complete','config':config}))


class ExperimentTests(unittest.TestCase):
    def test_spawn_loader_can_read_portable_memmaps(self):
        from us_pipeline.experiment import make_loader,settings_for
        from us_pipeline.dataset import ImageDataset
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);synthetic_prepared(root)
            loader=make_loader(ImageDataset(root,'train'),settings_for({'batch_size':4,'num_workers':2}))
            x,y,index=next(iter(loader))
            self.assertEqual(tuple(x.shape),(4,1,234,234))
            self.assertEqual(len(y),4);self.assertEqual(index.tolist(),[0,1,2,3])
            del loader

    def test_epoch_resume_reports_and_single_checkpoint_test_lock(self):
        import torch
        from us_pipeline.experiment import train_experiment,final_test
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);prepared=root/'prepared';prepared.mkdir();synthetic_prepared(prepared)
            settings={'epochs':1,'batch_size':4,'num_workers':0,'seed':4,'patience_epochs':5}
            report=train_experiment(prepared,root/'run',settings,device='cpu',progress=False)
            self.assertEqual(report['completed_epochs'],1)
            self.assertFalse(report['test_evaluated'])
            report=train_experiment(prepared,root/'run',{**settings,'epochs':2},device='cpu',resume=True,progress=False)
            self.assertEqual(report['completed_epochs'],2)
            result=final_test(prepared,root/'run',device='cpu',progress=False)
            self.assertEqual(result['metrics']['samples'],8)
            self.assertTrue((root/'run/test/evaluation.png').exists())
            self.assertIn('median_batch_ms',result['model_only_benchmarks'][0])
            same=final_test(prepared,root/'run',device='cpu',progress=False)
            self.assertEqual(result['checkpoint_sha256'],same['checkpoint_sha256'])
            with self.assertRaisesRegex(ValueError,'already evaluated test'):
                train_experiment(prepared,root/'run',{**settings,'epochs':3},device='cpu',resume=True,progress=False)

    def test_smoke_cannot_be_presented_as_formal_test(self):
        import torch
        from us_pipeline.experiment import train_experiment,final_test
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'prepared';p.mkdir();synthetic_prepared(p)
            train_experiment(p,root/'smoke',{'epochs':1,'batch_size':4,'num_workers':0,'smoke':True},device='cpu',progress=False)
            self.assertTrue(json.loads((root/'smoke/validation/metrics.json').read_text())['smoke_only'])
            with self.assertRaisesRegex(ValueError,'smoke checkpoint'):
                final_test(p,root/'smoke',device='cpu',progress=False)


if __name__=='__main__':
    unittest.main()
