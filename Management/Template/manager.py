'''
manager template.
'''

__all__=['manager']

manager_template="""\
import mkl
from HamiltonianPy import *
from source import *
from collections import OrderedDict

if __name__=='__main__':
    # Forbid multithreading
    mkl.set_num_threads(1)

    # When using log files, set it to be False
    Engine.DEBUG=True

    # When using log files and data files, it's safe to set it to be True.
    Engine.MKDIR=False

    # parameters
    parameters=OrderedDict()
"""

def manager():
    return manager_template
