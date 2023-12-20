from unittest import TestCase

import test_config
from lstm_predictor.core import config
from lstm_predictor.util.price_reader import read_prices


class Test(TestCase):

    def test_read_prices(self):

        def get_shape(week):
            return read_prices(test_config.prices_path, week, 4).shape[0]

        self.assertEqual(get_shape(0), 89)
        self.assertEqual(get_shape(1), 96)
        self.assertEqual(get_shape(2), 103)
        self.assertEqual(get_shape(3), 110)
        self.assertEqual(get_shape(4), 117)

    def test_read_prices_prod_file(self):

        def get_shape(week):
            return read_prices(f'../../{config.prices_path}', week, 104).shape[0]

        self.assertEqual(get_shape(0), 1461)
        self.assertEqual(get_shape(1), 1468)
        self.assertEqual(get_shape(2), 1475)
        self.assertEqual(get_shape(3), 1482)
        self.assertEqual(get_shape(103), 2182)
        self.assertEqual(get_shape(104), 2189)
