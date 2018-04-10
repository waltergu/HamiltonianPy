'''
CDMFT test.
'''

__all__=['cdmft']

from unittest import TestCase,TestLoader,TestSuite

class TestCDMFT(TestCase):
    pass

cdmft=TestSuite([
            TestLoader().loadTestsFromTestCase(TestCDMFT),
            ])

if __name__=='__main__':
    from unittest import main
    main(verbosity=2)
