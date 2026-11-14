"""Execute every code cell in local CPU smoke mode; never evaluate test."""
import contextlib
import json
import os
from pathlib import Path
import tempfile
import time
import nbformat

ROOT=Path(__file__).resolve().parent.parent
qa=Path(tempfile.mkdtemp(prefix='repair_',dir=ROOT/'colab/_local_validation'))
demo=qa/'Demo';demo.mkdir()
os.link(ROOT/'colab/cnn78u_colab_bundle.zip',demo/'cnn78u_colab_bundle.zip')
os.environ['CNN78U_LOCAL_NOTEBOOK_TEST']='1'
os.environ['CNN78U_LOCAL_DEMO']=str(demo)
os.environ['MPLCONFIGDIR']=str(qa/'mpl')
nb=nbformat.read(ROOT/'colab/CNN78u_Training_Repair_20pct.ipynb',as_version=4)
nbformat.validate(nb)
namespace={}; started=time.perf_counter();executed=0
with (qa/'execution.log').open('w') as log, contextlib.redirect_stdout(log):
    for i,cell in enumerate(nb.cells):
        if cell.cell_type=='code':
            exec(compile(cell.source,f'notebook_cell_{i}','exec'),namespace)
            executed+=1
    # Both arms must use identical initial weights and fixed subset.
    assert namespace['control']['initial_parameters_sha256']==namespace['candidate']['initial_parameters_sha256']
    assert namespace['control']['train_subset_sha256']==namespace['candidate']['train_subset_sha256']
    # A completed arm is cached; changed settings are refused.
    runner=namespace['repair']; path=namespace['LOCAL_REPAIR']/'wd_0'
    args=(namespace['PREPARED'],path,namespace['PERSISTENT']/'wd_0',namespace['indices'],namespace['probe_ids'])
    cached=runner.run_arm(*args,namespace['SETTINGS'],weight_decay=0,smoke=True)
    assert cached==namespace['candidate']
    try:
        runner.run_arm(*args,{**namespace['SETTINGS'],'learning_rate':.002},weight_decay=0,smoke=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Changed contract was accepted')
    # Simulate a disconnect just after the first completed epoch was persisted.
    from unittest.mock import patch
    resume_root=qa/'resume_run'; resume_persistent=qa/'resume_persistent'
    settings={**namespace['SETTINGS'],'epochs':2}
    original_save=runner.save_checkpoint
    def interrupt_after_epoch(path,value,persistent=None):
        original_save(path,value,persistent)
        if path.name=='last.pt' and value['epoch']==1:
            raise RuntimeError('Simulated disconnect after epoch checkpoint')
    resume_args=(namespace['PREPARED'],resume_root,resume_persistent,namespace['indices'],namespace['probe_ids'])
    try:
        with patch.object(runner,'save_checkpoint',interrupt_after_epoch):
            runner.run_arm(*resume_args,settings,weight_decay=0,smoke=True)
    except RuntimeError as exc:
        assert 'Simulated disconnect' in str(exc)
    else:
        raise AssertionError('Disconnect simulation did not execute')
    recovered=runner.run_arm(*resume_args,settings,weight_decay=0,smoke=True)
    assert [h['epoch'] for h in recovered['history']]==[1,2]
    assert recovered['steps']==2*namespace['candidate']['steps']
summary=json.loads((namespace['PERSISTENT']/'experiment_summary.json').read_text())
assert summary['smoke_only'] and not summary['test_evaluated']
assert not list(namespace['LOCAL_REPAIR'].glob('*/test'))
result={'status':'passed','code_cells_executed':executed,'seconds':time.perf_counter()-started,
        'mode':'local CPU smoke; not Colab GPU training','test_evaluated':False,'qa_directory':str(qa),
        'checks':['same initialization','same subset','cached completed arm','changed contract rejected',
                  'epoch-boundary disconnect recovery','no test reports']}
(ROOT/'colab/repair_notebook_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
