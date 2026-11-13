import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from build_dynamic_universe import build_universe, ident, resolve_index, save


def member(day, index, permno=1, ticker='OLD', member_id='a'):
    return dict(date=day, index_name=index, permno=permno, ticker=ticker, member_id=member_id)


class UniverseTests(unittest.TestCase):
    def test_union_preserves_index_membership_and_ticker_changes(self):
        rows = [member('2020-01-03', i) for i in ('SP500', 'NDX100', 'DJIA30')]
        rows += [member('2020-01-04', i, ticker='NEW') for i in ('SP500', 'NDX100', 'DJIA30')]
        rows += [member('2020-01-04', 'NDX100', 2, 'BRK.B', 'b')]
        daily, long = build_universe(['2020-01-03', '2020-01-04'], rows)
        self.assertEqual(daily, {'2020-01-03': ['OLD'], '2020-01-04': ['BRK.B', 'NEW']})
        self.assertEqual(len(long), 3)
        self.assertEqual(long[0]['indices'], 'DJIA30|NDX100|SP500')

    def test_missing_index_or_date_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing historical'):
            build_universe(['2020-01-01'], [member('2020-01-01', 'SP500')])
        with self.assertRaises(ValueError):
            build_universe(['2020-01-01'], [])

    def test_unmapped_members_are_not_silently_dropped(self):
        for permno, ticker in [(None, 'A'), (float('nan'), 'A'), (1, None), (1, ' ')]:
            with self.subTest(permno=permno, ticker=ticker), self.assertRaises(ValueError):
                build_universe(['2020-01-01'], [member('2020-01-01', 'SP500', permno, ticker)])

    def test_ambiguous_link_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            build_universe(['2020-01-01'], [member('2020-01-01', 'NDX100', p, str(p)) for p in (1, 2)])

    def test_ticker_collision_rejected(self):
        with self.assertRaisesRegex(ValueError, 'multiple securities'):
            build_universe(['2020-01-01'], [member('2020-01-01', 'NDX100', p, 'A', str(p)) for p in (1, 2)])

    def test_index_selection_and_sql_identifier_safety(self):
        rows = [{'index_id': '123', 'name': 'NASDAQ-100'},
                {'index_id': '456', 'name': 'NASDAQ COMPOSITE'}]
        self.assertEqual(resolve_index(rows, 'NDX100', None), '123')
        with self.assertRaises(RuntimeError):
            resolve_index(rows, 'DJIA30', None)
        with self.assertRaises(ValueError):
            ident('foo; DROP TABLE bar')

    def test_output_roundtrip(self):
        rows = [member(dt.date(2020, 1, 1), i) for i in ('SP500', 'NDX100', 'DJIA30')]
        daily, long = build_universe(['2020-01-01'], rows)
        with tempfile.TemporaryDirectory() as folder:
            save(Path(folder), daily, long, {})
            self.assertEqual(json.loads((Path(folder)/'daily_tickers.json').read_text()), daily)
            manifest = json.loads((Path(folder)/'manifest.json').read_text())
            self.assertEqual(manifest['daily_counts']['2020-01-01']['union'], 1)
            self.assertEqual(manifest['daily_counts']['2020-01-01']['SP500'], 1)


if __name__ == '__main__':
    unittest.main()
