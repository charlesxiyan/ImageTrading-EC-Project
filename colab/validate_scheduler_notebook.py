"""Execute the notebook and verify recovery across an actual scheduled LR change."""
import contextlib
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
import nbformat
import torch

ROOT=Path(__file__).resolve().parent.parent
qa=Path(tempfile.mkdtemp(prefix='scheduler_',dir=ROOT/'colab/_local_validation'))
demo=qa/'Demo';demo.mkdir()
os.link(ROOT/'colab/cnn78u_colab_bundle.zip',demo/'cnn78u_colab_bundle.zip')
os.environ['CNN78U_LOCAL_NOTEBOOK_TEST']='1';os.environ['CNN78U_LOCAL_DEMO']=str(demo)
os.environ['MPLCONFIGDIR']=str(qa/'mpl')
nb=nbformat.read(ROOT/'colab/CNN78u_Scheduler_20pct.ipynb',as_version=4)
nbformat.validate(nb)
ns={};started=time.perf_counter();executed=0
with (qa/'execution.log').open('w') as log,contextlib.redirect_stdout(log):
    for i,c in enumerate(nb.cells):
        if c.cell_type=='code':
            exec(compile(c.source,f'cell_{i}','exec'),ns);executed+=1
    r=ns['runner'];settings={**ns['SETTINGS'],'epochs':3,'lr_patience':0}
    args=(ns['PREPARED'],qa/'resume',qa/'resume_drive',ns['indices'],ns['probe_ids'],settings)
    original_save=r.save_checkpoint;original_step=r.ScheduleControl.step
    def forced_plateau(self, metric):
        # Synthetic plateau tests schedule mechanics, not financial performance.
        return original_step(self,.7)
    def disconnect(path,value,persistent=None):
        original_save(path,value,persistent)
        if path.name=='last.pt' and value['epoch']==2:
            raise RuntimeError('Simulated disconnect after LR reduction')
    with patch.object(r.ScheduleControl,'step',forced_plateau):
        try:
            with patch.object(r,'save_checkpoint',disconnect):
                r.run_experiment(*args,weight_decay=0,smoke=True)
        except RuntimeError as exc:
            assert 'Simulated disconnect' in str(exc)
        else:
            raise AssertionError('Checkpoint interrupt did not run')
        recovered=r.run_experiment(*args,weight_decay=0,smoke=True)
        full_args=(ns['PREPARED'],qa/'full',qa/'full_drive',ns['indices'],ns['probe_ids'],settings)
        uninterrupted=r.run_experiment(*full_args,weight_decay=0,smoke=True)
    assert [h['learning_rate_used'] for h in recovered['history']]==[.001,.001,.0005]
    assert recovered['schedule']==uninterrupted['schedule']
    a=torch.load(qa/'resume/last.pt',map_location='cpu',weights_only=False)
    b=torch.load(qa/'full/last.pt',map_location='cpu',weights_only=False)
    for key in a['model']:
        torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    cached=r.run_experiment(*args,weight_decay=0,smoke=True)
    assert cached==recovered
    try:
        r.run_experiment(*args[:-1],{**settings,'min_lr':.0001},weight_decay=0,smoke=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Changed scheduler settings accepted')
assert not ns['result']['test_evaluated']
assert not list(ns['LOCAL_RUN'].glob('**/test'))
result=dict(status='passed',code_cells_executed=executed,seconds=time.perf_counter()-started,
    mode='Local CPU smoke and synthetic plateau; not full Colab training',test_evaluated=False,
    qa_directory=str(qa),checks=['all notebook cells','LR-reduction checkpoint recovery',
        'resumed model exactly equals uninterrupted CPU model','cached result','changed scheduler rejected'])
(ROOT/'colab/scheduler_notebook_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
