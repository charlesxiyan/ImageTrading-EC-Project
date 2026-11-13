"""Inspect WRDS sources for the six-month intraday demo; no market data download."""
import getpass
import json
import os
from pathlib import Path
from build_dynamic_universe import inspect, columns, table_ident, records

DEFAULT_OUTPUT = Path.home()/'Market Data'


def inspect_demo(db, output=DEFAULT_OUTPUT):
    output = Path(output)
    report = inspect(db)
    report['demo_request'] = {
        'start': '2023-01-01', 'end': '2023-06-30',
        'bar_minutes': 5, 'session': 'regular',
        'universe': ['SP500', 'NDX100', 'DJIA30'],
    }
    report['source_details'] = {}
    libraries = ['taqm_2023', 'wrdsapps_link_crsp_taqm', 'wrdsapps_link_crsp_taq']
    for library in libraries:
        try:
            tables = db.list_tables(library=library)
            report['source_details'][library] = {'tables': tables}
            if library == 'taqm_2023':
                candidates = sorted(t for t in tables if t.startswith('ctm_'))[:2]
            else:
                candidates = tables[:10]
            details = report['source_details'][library]
            details['columns'] = {}
            details['samples'] = {}
            for table in candidates:
                full_name = f'{library}.{table}'
                details['columns'][table] = sorted(columns(db, full_name))
                # Bounded samples establish actual read permission and field types.
                details['samples'][table] = records(
                    db, f'SELECT * FROM {table_ident(full_name)} LIMIT 2')
        except Exception as exc:
            report['errors'][library] = f'{type(exc).__name__}: {exc}'
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'wrds_demo_schema.json'
    destination.write_text(json.dumps(report, indent=2, default=str) + '\n', encoding='utf-8')
    print(f'Schema report saved to {destination}')
    return report


if __name__ == '__main__':
    import wrds
    connection = wrds.Connection(wrds_username=os.environ.get('WRDS_USERNAME') or getpass.getuser())
    try:
        inspect_demo(connection)
    finally:
        connection.close()
