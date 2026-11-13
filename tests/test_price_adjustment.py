import unittest
import numpy as np
import pandas as pd
from adjust_bar_prices import prepare_factors, adjust_frame


def bars(date='2023-01-03',ticker='OLD'):
    return pd.DataFrame([dict(date=date,permno=1,ticker=ticker,status='observed',
        open=100.,high=110.,low=90.,close=105.,vwap=102.,volume=1000,trade_count=10)])


class PriceAdjustmentTests(unittest.TestCase):
    def test_split_and_latest_anchor_unchanged(self):
        f,a=prepare_factors(pd.DataFrame({'date':['2023-01-03','2023-01-04'],
            'permno':[1,1],'cfacpr':[2.,1.]}),'2023-01-04')
        before=adjust_frame(bars(),f)
        after=adjust_frame(bars('2023-01-04','NEW'),f)
        self.assertEqual(before.open.iloc[0],50.)
        self.assertEqual(before.volume.iloc[0],1000)
        self.assertEqual(before.trade_count.iloc[0],10)
        self.assertEqual(before.vwap.iloc[0],51.)
        self.assertEqual(after.open.iloc[0],100.)
        self.assertEqual(after.ticker.iloc[0],'NEW')

    def test_nonunit_anchor_and_future_events_cancel(self):
        f,a=prepare_factors(pd.DataFrame({'date':['2023-01-03','2023-01-04','2024-01-01'],
            'permno':[1,1,1],'cfacpr':[6.,3.,1.]}),'2023-01-04')
        self.assertEqual(adjust_frame(bars(),f).open.iloc[0],50.)
        self.assertEqual(a.base_cfacpr.iloc[0],3.)
        self.assertEqual(a.adjustment_base_date.iloc[0],'2023-01-04')

    def test_reverse_split_can_increase_history(self):
        f,_=prepare_factors(pd.DataFrame({'date':['2023-01-03','2023-01-04'],
            'permno':[1,1],'cfacpr':[.5,1.]}),'2023-01-04')
        self.assertEqual(adjust_frame(bars(),f).open.iloc[0],200.)

    def test_observed_prices_require_exact_day_factor(self):
        f,_=prepare_factors(pd.DataFrame({'date':['2023-01-04'],'permno':[1],'cfacpr':[1.]}),'2023-01-04')
        with self.assertRaisesRegex(ValueError,'exact-day'):
            adjust_frame(bars(),f)

    def test_halted_missing_prices_are_preserved(self):
        b=bars()
        b.loc[0,['open','high','low','close','vwap']]=np.nan
        b.loc[0,'status']='no_symbol_mapping'
        b.loc[0,['volume','trade_count']]=0
        f,_=prepare_factors(pd.DataFrame({'date':['2023-01-04'],'permno':[1],'cfacpr':[1.]}),'2023-01-04')
        a=adjust_frame(b,f)
        self.assertTrue(a.close.isna().all())
        self.assertEqual(a.adjustment_status.iloc[0],'missing_factor_no_price')

    def test_invalid_or_duplicate_factors_fail(self):
        for factor in [0.,-1.,float('nan'),float('inf')]:
            with self.subTest(factor=factor),self.assertRaises(ValueError):
                prepare_factors(pd.DataFrame({'date':['2023-01-03'],'permno':[1],'cfacpr':[factor]}),'2023-01-03')
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            prepare_factors(pd.DataFrame({'date':['2023-01-03']*2,'permno':[1,1],'cfacpr':[1.,2.]}),'2023-01-03')

    def test_double_adjustment_rejected(self):
        f,_=prepare_factors(pd.DataFrame({'date':['2023-01-03'],'permno':[1],'cfacpr':[1.]}),'2023-01-03')
        adjusted=adjust_frame(bars(),f)
        with self.assertRaisesRegex(ValueError,'already'):
            adjust_frame(adjusted,f)


if __name__=='__main__':
    unittest.main()
