'''
CDMFTCCT test.
'''

__all__=['cdmftcct']

from unittest import TestCase,TestLoader,TestSuite

class TestCDMFTCCT(TestCase):
    pass

cdmftcct=TestSuite([
            TestLoader().loadTestsFromTestCase(TestCDMFTCCT),
            ])

if __name__=='__main__':
    from unittest import main
    main(verbosity=2)
