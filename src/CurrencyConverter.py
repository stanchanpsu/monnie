class CurrencyConverter(object):
    conversions = {
        'EUR': 0.86,
        'USD': 1.00,
        'RMB': 6.51
    }

    @classmethod
    def convert_currency(cls, amount, to_currency):
        return str(amount * cls.conversions[to_currency]) + ' ' + to_currency