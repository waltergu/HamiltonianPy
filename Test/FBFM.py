'''
FBFM test.
'''

__all__=['test_fbfm']

def test_fbfm(arg):
    if arg in ('fbfm','all'):
        from HamiltonianPy.FBFM import test
        test.test_fbfm()
