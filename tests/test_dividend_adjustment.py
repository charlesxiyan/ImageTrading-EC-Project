import unittest
import numpy as np
import pandas as pd
from adjust_dividend_bars import prepare_adjustments,apply_adjustments


def inputs(split=False):
    d=['2023-01-03','2023-01-04','2023-01-05']
    f=pd.DataFrame({'permno':[1]*3,'date':d,'cfacpr':[2.,1.,1.] if split else [1.]*3,
                    'cfacshr':[2.,1.,1.] if split else [1.]*3})
    r=f.copy();r['prc']=[100.,50. if split else 100.,100.]
    e=pd.DataFrame([dict(permno=1,distcd=1232,divamt=1. if split else 5.,
                        facpr=0.,facshr=0.,exdt='2023-01-04')])
    return f,e,r


def bar(date='2023-01-03'):
    return pd.DataFrame([dict(date=date,permno=1,ticker='OLD',status='observed',
        open=100.,high=110.,low=90.,close=105.,vwap=102.,volume=100,trade_count=3)])


class DividendTests(unittest.TestCase):
    def test_cash_changes_prices_not_volume_or_exdate(self):
        f,e,r=inputs();f,a,audit,cash=prepare_adjustments(f,e,r)
        before=apply_adjustments(bar(),f)
        self.assertEqual(before.open.iloc[0],95.)
        self.assertEqual(before.volume.iloc[0],100.)
        self.assertEqual(before.volume_raw.iloc[0],100)
        self.assertEqual(before.open_raw.iloc[0],100.)
        self.assertEqual(before.trade_count.iloc[0],3)
        self.assertEqual(apply_adjustments(bar('2023-01-04'),f).open.iloc[0],100.)

    def test_same_day_split_uses_exdate_share_units(self):
        f,e,r=inputs(True);f,a,audit,cash=prepare_adjustments(f,e,r)
        self.assertEqual(cash.previous_close_on_exdate_basis.iloc[0],50.)
        self.assertAlmostEqual(cash.cash_multiplier.iloc[0],.98)
        adjusted=apply_adjustments(bar(),f)
        self.assertAlmostEqual(adjusted.open.iloc[0],49.)
        self.assertEqual(adjusted.volume.iloc[0],200.)

    def test_multiple_cash_payments_on_same_exdate_are_summed(self):
        f,e,r=inputs();e=pd.concat([e,e.assign(distcd=1272,divamt=2.)],ignore_index=True)
        f,a,audit,cash=prepare_adjustments(f,e,r)
        self.assertAlmostEqual(cash.cash_multiplier.iloc[0],.93)
        self.assertEqual(len(cash),1)
        self.assertEqual(cash.distribution_records.iloc[0],2)

    def test_successive_dividends_compound(self):
        f,e,r=inputs();e=pd.concat([e,e.assign(exdt='2023-01-05')],ignore_index=True)
        f,a,audit,cash=prepare_adjustments(f,e,r)
        self.assertAlmostEqual(apply_adjustments(bar(),f).open.iloc[0],90.25)
        self.assertAlmostEqual(apply_adjustments(bar('2023-01-04'),f).open.iloc[0],95.)

    def test_events_after_security_anchor_do_not_adjust_history(self):
        f,e,r=inputs();f=f.iloc[:2];e=e.assign(exdt='2023-01-05')
        f,a,audit,cash=prepare_adjustments(f,e,r)
        self.assertEqual(audit.treatment.iloc[0],'outside_adjustable_history')
        self.assertEqual(apply_adjustments(bar(),f).open.iloc[0],100.)

    def test_unsupported_or_already_factored_cash_fails(self):
        for patch in [dict(distcd=9999),dict(facpr=.1),dict(divamt=200.)]:
            f,e,r=inputs()
            with self.subTest(patch=patch),self.assertRaises(ValueError):
                prepare_adjustments(f,e.assign(**patch),r)

    def test_distribution_with_existing_factor_is_not_double_applied(self):
        f,e,r=inputs(True);e=e.assign(distcd=5523,facpr=1.,facshr=1.,divamt=0.)
        f,a,audit,cash=prepare_adjustments(f,e,r)
        self.assertEqual(audit.treatment.iloc[0],'covered_by_cfacpr_cfacshr')
        self.assertEqual(apply_adjustments(bar(),f).open.iloc[0],50.)

    def test_missing_prices_and_zero_volume_stay_missing_and_zero(self):
        f,e,r=inputs();f,a,audit,cash=prepare_adjustments(f,e,r)
        b=bar('2023-01-06');b.loc[0,['open','high','low','close','vwap']]=np.nan
        b.loc[0,'status']='no_symbol_mapping';b.loc[0,['volume','trade_count']]=0
        adjusted=apply_adjustments(b,f)
        self.assertTrue(adjusted.close.isna().all())
        self.assertEqual(adjusted.volume.iloc[0],0.)
        with self.assertRaisesRegex(ValueError,'exact-day'):
            apply_adjustments(bar('2023-01-06'),f)

    def test_reference_vintage_mismatch_and_double_adjustment_fail(self):
        f,e,r=inputs();bad=r.copy();bad.loc[0,'cfacpr']=3.
        with self.assertRaisesRegex(ValueError,'vintage'):
            prepare_adjustments(f,e,bad)
        f,a,audit,cash=prepare_adjustments(f,e,r)
        with self.assertRaisesRegex(ValueError,'double'):
            apply_adjustments(apply_adjustments(bar(),f),f)


if __name__=='__main__':
    unittest.main()
