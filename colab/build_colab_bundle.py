"""Package only portable code and current prepared next-bar data for manual upload.

    python colab/build_colab_bundle.py                       # v1: artifacts/cnn78u_5min_h1
    python colab/build_colab_bundle.py --version v2          # v2: artifacts/cnn78u_5min_h1_v2,
                                                             # all windows kept (configs/us_5min_v2.json)
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
VERSIONS={'v1':('artifacts/cnn78u_5min_h1/prepared','configs/us_5min.json','cnn78u_colab_bundle.zip'),
          'v2':('artifacts/cnn78u_5min_h1_v2/prepared','configs/us_5min_v2.json','cnn78u_colab_bundle_v2.zip')}


def build(version='v1',prepared=None,output=None):
    rel_prepared,config,name=VERSIONS[version]
    PREPARED=Path(prepared) if prepared else ROOT/rel_prepared
    OUTPUT=Path(output) if output else ROOT/'colab'/name
    m=json.loads((PREPARED/'manifest.json').read_text())
    if m['status']!='complete' or m['config']['lookback']!=78 or m['config']['horizon']!=1:
        raise ValueError('Prepare the current CNN78u next-bar data first')
    audit=json.loads((PREPARED/'validation_report.json').read_text())
    if audit['status']!='validated_complete':
        raise ValueError('Prepared sample audit is missing')
    paths={f'us_pipeline/{p.name}':p for p in sorted((ROOT/'us_pipeline').glob('*.py'))}
    paths.update({f'prepared/{p.name}':p for p in sorted(PREPARED.iterdir()) if p.is_file()})
    paths.update({'configs/'+Path(config).name:ROOT/config,
        'requirements-colab.txt':ROOT/'requirements-colab.txt','README.md':ROOT/'colab/README.md',
        'US_PIPELINE.md':ROOT/'US_PIPELINE.md'})
    report={'format_version':1,'bundle_version':version,'task':'CNN78u, L=78, H=1, five-minute bars',
        'drop_zero_returns':m['config'].get('drop_zero_returns',True),
        'samples':{k:v['samples'] for k,v in m['splits'].items()},'files':{}}
    temporary=OUTPUT.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for relative,path in paths.items():
            report['files'][relative]={'bytes':path.stat().st_size,
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            archive.write(path,relative,compress_type=zipfile.ZIP_STORED if path.suffix=='.parquet' else zipfile.ZIP_DEFLATED)
        archive.writestr('bundle_manifest.json',json.dumps(report,indent=2)+'\n')
    temporary.replace(OUTPUT)
    summary={'archive':str(OUTPUT),'bytes':OUTPUT.stat().st_size,
        'sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),'files':len(paths)+1,'samples':report['samples']}
    (OUTPUT.parent/('upload_manifest.json' if version=='v1' else f'upload_manifest_{version}.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--version',choices=sorted(VERSIONS),default='v1')
    a.add_argument('--prepared',help='override the prepared/ folder')
    a.add_argument('--output',help='override the output zip path')
    a=a.parse_args()
    build(a.version,a.prepared,a.output)
