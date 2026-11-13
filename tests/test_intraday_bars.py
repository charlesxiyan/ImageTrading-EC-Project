import unittest
import pandas as pd
from download_intraday_bars import aggregate_sql, complete_grid


class BarTests(unittest.TestCase):
    def setUp(self):
        self.day='2023-03-13'
        self.universe=pd.DataFrame([
            dict(date=self.day,permno=1,ticker='AAA',indices='SP500',identifier_status='date_effective'),
            dict(date=self.day,permno=2,ticker='HALT',indices='NDX100',identifier_status='last_known_inactive')])
        self.mapping=pd.DataFrame([dict(date=self.day,permno=1,ticker='AAA',sym_root='AAA',sym_suffix='',match_lvl=1)])
        self.bars=pd.DataFrame([dict(sym_root='AAA',sym_suffix='',bucket=0,open=10.,high=12.,low=9.,close=11.,volume=100,trade_count=2,vwap=10.5)])

    def test_grid_preserves_halted_and_empty_intervals(self):
        result,q=complete_grid(self.day,self.universe,self.mapping,self.bars,5)
        self.assertEqual(len(result),156)
        self.assertEqual(q['observed_rows'],1)
        self.assertEqual(q['unmapped_members'],1)
        self.assertEqual(q['members_with_no_eligible_trades'],[2])
        self.assertEqual(result[result.permno.eq(2)].status.unique().tolist(),['no_symbol_mapping'])
        self.assertTrue(result[result.status.ne('observed')].close.isna().all())
        self.assertEqual(result.volume.sum(),100)

    def test_timezone_after_daylight_saving(self):
        result,_=complete_grid(self.day,self.universe,self.mapping,self.bars,5)
        self.assertEqual(str(result.bar_start.iloc[0]),'2023-03-13 09:30:00-04:00')
        self.assertEqual(result.bar_end.iloc[77].strftime('%H:%M'),'16:00')

    def test_missing_active_mapping_fails(self):
        self.universe.loc[1,'identifier_status']='date_effective'
        with self.assertRaisesRegex(RuntimeError,'lack TAQ mapping'):
            complete_grid(self.day,self.universe,self.mapping,self.bars,5)

    def test_invalid_ohlc_fails(self):
        self.bars.loc[0,'high']=8
        with self.assertRaisesRegex(RuntimeError,'invalid OHLC'):
            complete_grid(self.day,self.universe,self.mapping,self.bars,5)

    def test_query_is_bounded_and_identifiers_validated(self):
        sql=aggregate_sql('taqm_2023.ctm_20230103',5)
        self.assertIn("time_m < '16:00:00'",sql)
        self.assertIn("tr_corr='00'",sql)
        self.assertIn("translate(tr_scond, ' @EF', '') = ''",sql)
        with self.assertRaises(ValueError):
            aggregate_sql('taqm_2023.x;drop',5)
        with self.assertRaises(ValueError):
            aggregate_sql('taqm_2023.ctm_20230103',7)


class DownloadResumeTests(unittest.TestCase):
    def test_parallel_download_and_checksum_resume(self):
        import argparse
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from download_intraday_bars import download

        days=['2023-01-03','2023-01-04']
        universe=pd.DataFrame([dict(date=day,permno=1,ticker='AAA',indices='DJIA30|NDX100|SP500',
                                   identifier_status='date_effective') for day in days])
        mapping=pd.DataFrame([dict(date=day,permno=1,ticker='AAA',sym_root='AAA',sym_suffix='',
                                  match_lvl=1.,mapping_source='wrds_daily_link') for day in days])
        calls=[]

        class FakeDB:
            def raw_sql(self,sql,params=None):
                if 'set_config' in sql:
                    return pd.DataFrame({'set_config':['900000']})
                if 'cfacpr' in sql:
                    return pd.DataFrame([dict(date=day,permno=1,cfacpr=1.,cfacshr=1.,prc=11.,vol=100) for day in days])
                calls.append(sql)
                return pd.DataFrame([dict(sym_root='AAA',sym_suffix='',bucket=0,open=10.,high=12.,
                    low=9.,close=11.,volume=100,trade_count=2,vwap=10.5)])
            def close(self):
                pass

        with tempfile.TemporaryDirectory() as folder:
            args=argparse.Namespace(start=days[0],end=days[-1],minutes=5,workers=2,
                                    username='test',limit_days=None,output=Path(folder))
            with patch('download_intraday_bars.historical_universe',return_value=(days,{d:['AAA'] for d in days},universe,{})), \
                 patch('download_intraday_bars.get_mapping',return_value=mapping), \
                 patch('wrds.Connection',side_effect=lambda **kwargs:FakeDB()):
                download(FakeDB(),args)
                self.assertEqual(len(calls),2)
                manifest=json.loads((Path(folder)/'manifest.json').read_text())
                self.assertEqual(manifest['status'],'complete')
                self.assertEqual(manifest['total_rows'],156)
                from validate_intraday_demo import validate
                summary=validate(Path(folder))
                self.assertEqual(summary['status'],'validated_complete')
                self.assertEqual(summary['observed_rows'],2)
                self.assertEqual(summary['empty_rows'],154)
                download(FakeDB(),args)
                self.assertEqual(len(calls),2,'Verified completed files should not be downloaded again')
                # A corrupted file must be recomputed even if the manifest says complete.
                (Path(folder)/'bars_5min'/f'{days[0]}.parquet').write_bytes(b'corrupted')
                download(FakeDB(),args)
                self.assertEqual(len(calls),3)


if __name__=='__main__':
    unittest.main()
