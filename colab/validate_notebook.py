"""Execute the CPU smoke branch locally; no Google Drive or real test inference."""
import hashlib
import json
import os
from pathlib import Path
import time
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]


def validate():
    qa=ROOT/'colab/_local_validation'/time.strftime('%Y%m%d_%H%M%S')
    demo=qa/'Demo';demo.mkdir(parents=True)
    (demo/'cnn78u_colab_bundle.zip').hardlink_to(ROOT/'colab/cnn78u_colab_bundle.zip')
    for name in ['runtime','ipython','mpl','jupyter_config','jupyter/kernels/cnn78u_local']:
        (qa/name).mkdir(parents=True,exist_ok=True)
    kernel={'argv':[str(ROOT/'.venv/bin/python'),'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'CNN78u local smoke QA','language':'python'}
    (qa/'jupyter/kernels/cnn78u_local/kernel.json').write_text(json.dumps(kernel))
    os.environ.update({'CNN78U_LOCAL_NOTEBOOK_TEST':'1','CNN78U_LOCAL_DEMO':str(demo),
        'JUPYTER_PATH':str(qa/'jupyter'),'JUPYTER_RUNTIME_DIR':str(qa/'runtime'),
        'JUPYTER_CONFIG_DIR':str(qa/'jupyter_config'),'IPYTHONDIR':str(qa/'ipython'),'MPLCONFIGDIR':str(qa/'mpl')})
    notebook=nbformat.read(ROOT/'colab/CNN78u_Colab_Baseline.ipynb',as_version=4)
    started=time.perf_counter()
    try:
        NotebookClient(notebook,timeout=300,kernel_name='cnn78u_local',
            resources={'metadata':{'path':str(ROOT)}}).execute()
    finally:
        nbformat.write(notebook,qa/'executed_smoke.ipynb')
    result=demo/'runs/local_smoke'
    summary=json.loads((result/'summary.json').read_text())
    assert summary['mode']=='smoke_only' and summary['test_evaluated'] is False
    assert not (result/'test/metrics.json').exists()
    assert (result/'validation/evaluation.png').exists() and (result/'hardware_benchmark.json').exists()
    report={'status':'passed','execution_seconds':time.perf_counter()-started,
        'bundle_sha256':hashlib.sha256((ROOT/'colab/cnn78u_colab_bundle.zip').read_bytes()).hexdigest(),
        'code_cells_executed':sum(c.cell_type=='code' and c.execution_count is not None for c in notebook.cells),
        'runtime':'local CPU smoke only, not Colab GPU','real_test_model_evaluation':False,'qa_directory':str(qa)}
    (ROOT/'colab/notebook_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    validate()
