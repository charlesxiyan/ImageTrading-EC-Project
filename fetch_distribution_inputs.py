"""Fetch auditable CRSP distributions and daily references for local bar adjustment."""
import argparse
import getpass
import os
import json
from pathlib import Path

import pandas as pd
import psycopg2


def fetch(root, output):
    if output.exists():
        raise ValueError('Use a new output directory')
    universe=pd.read_parquet(root/'daily_universe.parquet')
    ids=sorted(int(x) for x in universe.permno.unique())
    start=universe.date.min()
    end=universe.date.max()
    connection=psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu',port=9737,
        dbname='wrds',user=os.environ.get('WRDS_USERNAME') or getpass.getuser(),connect_timeout=30,sslmode='require')
    try:
        with connection.cursor() as cursor:
            cursor.execute('''SELECT permno,distcd,divamt,facpr,facshr,dclrdt,exdt,rcrddt,paydt
                FROM crsp_a_stock.dsedist WHERE permno=ANY(%s)
                AND exdt BETWEEN %s AND %s ORDER BY permno,exdt,distcd''',(ids,start,end))
            events=pd.DataFrame(cursor.fetchall(),columns=[c.name for c in cursor.description])
            cursor.execute('''SELECT permno,date,prc,cfacpr,cfacshr FROM crsp_a_stock.dsf
                WHERE permno=ANY(%s) AND date BETWEEN %s::date-interval '60 days' AND %s::date
                ORDER BY permno,date''',(ids,start,end))
            daily=pd.DataFrame(cursor.fetchall(),columns=[c.name for c in cursor.description])
    finally:
        connection.close()
    for frame,cols in [(events,['divamt','facpr','facshr']),(daily,['prc','cfacpr','cfacshr'])]:
        for col in cols:
            frame[col]=pd.to_numeric(frame[col],errors='raise').astype(float)
    output.mkdir(parents=True)
    events.to_parquet(output/'distributions.parquet',index=False,compression='zstd')
    daily.to_parquet(output/'daily_references.parquet',index=False,compression='zstd')
    report={'start':str(start),'end':str(end),'permnos':len(ids),'events':len(events),
        'daily_rows':len(daily),'source_events':'crsp_a_stock.dsedist','source_daily':'crsp_a_stock.dsf',
        'fetched_at_utc':pd.Timestamp.now(tz='UTC').isoformat(),
        'event_code_counts':{str(k):int(v) for k,v in events.distcd.value_counts().items()}}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.home()/'Market Data/demo_2023_h1')
    p.add_argument('--output',type=Path,default=Path.home()/'Market Data/demo_2023_h1_distribution_inputs')
    a=p.parse_args()
    fetch(a.root,a.output)
