'''
-------------
Operator pack
-------------

Operator pack, including:
    * functions: fspoperators, JWBosonization
'''

__all__=['fspoperators','JWBosonization']

from ..FermionicPackage import FLinear,CREATION
from ..SpinPackage import SOperator,SpinMatrix
from collections import OrderedDict
from HamiltonianPy.Misc import parity
import numpy as np

def fspoperators(table,lattice):
    '''
    Generate the fermionic single particle operators corresponding to a table.

    Parameters
    ----------
    table : Table
        The index-sequence table of the fermionic single particle operators.
    lattice : Lattice
        The lattice on which the fermionic single particle operators are defined.

    Returns
    -------
    list of FOperator
        The fermionic single particle operators corresponding to the table.
    '''
    result=[]
    for ndx in sorted(table,key=table.get):
        result.append(FLinear(1,index=ndx,seq=table[ndx],rcoord=lattice.rcoord(ndx.pid),icoord=lattice.icoord(ndx.pid)))
    return result

def JWBosonization(operator,table):
    '''
    Convert an fermionic operator to a spin operator through the Jordan-Wigner transformation.

    Parameters
    ----------
    operator : FOperator
        The fermionic operator to be transformed.
    table : Table
        The index-sequence table of the fermions.

    Returns
    -------
    SOperator
        The Jordan-Wigner transformed spin operator.

    Notes
    -----
    The rules for the tag of the transformed operator:
        * '+' for creation operator
        * '-' for annihilation operator
        * 's' for sign operator
        * 'i' for identity operator
        * multi-characters for operator multiplications
    '''
    length=len(operator.indices)
    assert length%2==0
    dtype=np.array(operator.value).dtype
    permutation=sorted(range(length),key=lambda k:table[operator.indices[k].replace(nambu=None)])
    ms,counts,tags,inds=OrderedDict(),[],[],[]
    for k in permutation:
        leaf=table[operator.indices[k].replace(nambu=None)]
        m=np.array([[0.0,0.0],[1.0,0.0]],dtype=dtype) if operator.indices[k].nambu==CREATION else np.array([[0.0,1.0],[0.0,0.0]],dtype=dtype)
        tag='+' if operator.indices[k].nambu==CREATION else '-'
        if leaf in ms:
            ms[leaf]=ms[leaf].dot(m)
            counts[-1]+=1
            tags[-1]+=tag
        else:
            ms[leaf]=m
            counts.append(1)
            tags.append(tag)
            inds.append(operator.indices[k].replace(nambu=None))
    indices,sms,keys=[],[],ms.keys()
    TABLE,sign=table.reversed_table,np.array([[1.0,0.0],[0.0,-1.0]],dtype=dtype)
    for leaf in xrange(keys[0],keys[-1]+1):
        if leaf in ms:
            assert counts[0] in (1,2)
            length-=counts.pop(0)
            indices.append(inds.pop(0))
            sms.append(SpinMatrix(0.5,tags.pop(0)+('' if length%2==0 else 's'),matrix=ms[leaf] if length%2==0 else ms[leaf].dot(sign),dtype=dtype))
        elif length%2!=0:
            indices.append(TABLE[leaf].replace(nambu=None))
            sms.append(SpinMatrix(0.5,'s',matrix=sign,dtype=dtype))
    return SOperator(value=operator.value*parity(permutation),indices=indices,spins=sms)
