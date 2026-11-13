"""Read back dividend-adjusted bars and verify them against unchanged raw files."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from adjust_bar_prices import PRICE_COLUMNS,sha256


def validate(root):
    m=json.loads((root/'manifest.json').read_text())
    if m['status']!='complete':
        raise ValueError('Adjusted dataset is not complete')
    raw=Path(m['raw_root'])
    source=json.loads((raw/'manifest.json').read_text())
    if sha256(raw/'manifest.json')!=m['source_manifest_sha256']:
        raise ValueError('Raw manifest changed')
    rows=empty=0
    volume_ids=set()
    for day,entry in sorted(m['completed'].items()):
        path=root/entry['file'];source_path=raw/source['completed'][day]['file']
        assert sha256(path)==entry['sha256'],day
        assert sha256(source_path)==entry['raw_sha256'],day
        a=pd.read_parquet(path);r=pd.read_parquet(source_path)
        assert len(a)==len(r)==entry['rows'],day
        for col in PRICE_COLUMNS:
            np.testing.assert_allclose(a[col+'_raw'].astype(float),r[col].astype(float),equal_nan=True)
            np.testing.assert_allclose(a[col].astype(float),
                (r[col]*a.price_adjustment_multiplier).astype(float),equal_nan=True,rtol=1e-12)
            assert a[col].isna().equals(r[col].isna()),day
        np.testing.assert_array_equal(a.volume_raw,r.volume)
        expected=(r.volume.astype(float)*a.volume_adjustment_multiplier).fillna(0.)
        np.testing.assert_allclose(a.volume,expected,rtol=1e-12)
        for col in r.columns.difference(PRICE_COLUMNS+['volume']):
            pd.testing.assert_series_equal(a[col],r[col],check_names=False)
        anchor=a.date.eq(a.adjustment_base_date)
        assert np.isclose(a.loc[anchor,'price_adjustment_multiplier'],1.).all(),day
        assert np.isclose(a.loc[anchor,'volume_adjustment_multiplier'],1.).all(),day
        changed=~np.isclose(a.volume,a.volume_raw)
        volume_ids.update(int(x) for x in a.loc[changed,'permno'].unique())
        rows+=len(a);empty+=int(a.close.isna().sum())
    assert len(m['completed'])==len(source['completed'])==m['days']
    assert rows==m['rows']
    cash=pd.read_parquet(root/'cash_dividend_factors.parquet')
    report={'status':'validated_complete','files':len(m['completed']),'rows':rows,
        'missing_close_rows':empty,'cash_exdate_groups':len(cash),
        'cash_affected_permnos':int(cash.permno.nunique()),
        'cash_changed_rows':m['cash_changed_rows'],'volume_changed_rows':m['volume_changed_rows'],
        'volume_changed_permnos':sorted(volume_ids),
        'quoted_previous_closes':int(cash.previous_price_was_quote.sum()),
        'folder_bytes_before_report':sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
        'checks':['all output checksums','unchanged raw file checksums','raw field preservation',
                  'all price formulas','share volume formula','unchanged trade counts and identifiers',
                  'unchanged missing price masks','unchanged anchor-date prices and share units']}
    (root/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.home()/'Market Data/demo_2023_h1_dividend_adjusted')
    validate(p.parse_args().root)
