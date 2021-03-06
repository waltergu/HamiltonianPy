'''
config template.
'''

__all__=['config']

config_template="""\
from HamiltonianPy import *

__all__=['name','nnb','parametermap','idfmap','ttype','qnsmap']

# The configs of the model
name=None
nnb=None

# parametermap
parametermap=None

# idfmap
idfmap=lambda pid: None

# qnsmap
ttype='D'
qnsmap=lambda index: None

# terms
# example
# t=lambda statistics,**parameters: Hopping('t',parameters['t'],neighbour=1,statistics=statistics)
"""

def config():
    return config_template
