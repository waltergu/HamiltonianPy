'''
-------------
Operator pack
-------------

Operator pack, including:
    * functions: fspoperators, JWBosonization
'''

__all__=['fspoperators','JWBosonization','twisttransformation']

from ..Geometry import isparallel
from ..FockPackage import FockOperator,FOperator,FLinear,BLinear,CREATION,ANNIHILATION
from ..SpinPackage import SOperator,SpinMatrix
from ..Utilities import parity,RZERO
from collections import OrderedDict
import numpy as np

def fspoperators(table,lattice,statistics='f'):
    '''
    Generate single particle operators on the Fock space corresponding to a table.

    Parameters
    ----------
    table : Table
        The index-sequence table of the single particle operators.
    lattice : Lattice
        The lattice on which the single particle operators are defined.
    statistics : 'f','b'
        'f' for fermionic and 'b' for bosonic.

    Returns
    -------
    list of FOperator/BOperator
        The single particle operators on the Fock space corresponding to the table.
    '''
    assert statistics in ('f','b')
    result,CONSTRUCTOR=[],FLinear if statistics=='f' else BLinear
    for ndx in sorted(table,key=table.get):
        result.append(CONSTRUCTOR(1,index=ndx,seq=table[ndx],rcoord=lattice.rcoord(ndx.pid),icoord=lattice.icoord(ndx.pid)))
    return result

def JWBosonization(operator,table):
    '''
    Convert a fermionic/hard-core-bosonic operator to a spin operator through the Jordan-Wigner transformation.

    Parameters
    ----------
    operator : FOperator/BOperator
        The fermionic/hard-core-bosonic operator to be transformed.
    table : Table
        The index-sequence table of the fermions/hard-core-bosons.

    Returns
    -------
    SOperator
        The Jordan-Wigner transformed spin operator.

    Notes
    -----
    The rules for the tag of the transformed operator:
        * 'p' for creation operator
        * 'm' for annihilation operator
        * 's' for sign operator
        * 'i' for identity operator
        * multi-characters for operator multiplications
    '''
    length=len(operator.indices)
    assert length%2==0
    dtype=np.array(operator.value).dtype
    permutation=sorted(list(range(length)),key=lambda k:table[operator.indices[k].replace(nambu=None)])
    ms,counts,tags,inds=OrderedDict(),[],[],[]
    for k in permutation:
        leaf=table[operator.indices[k].replace(nambu=None)]
        m=np.array([[0.0,0.0],[1.0,0.0]],dtype=dtype) if operator.indices[k].nambu==CREATION else np.array([[0.0,1.0],[0.0,0.0]],dtype=dtype)
        tag='p' if operator.indices[k].nambu==CREATION else 'm'
        if leaf in ms:
            ms[leaf]=ms[leaf].dot(m)
            counts[-1]+=1
            tags[-1]+=tag
        else:
            ms[leaf]=m
            counts.append(1)
            tags.append(tag)
            inds.append(operator.indices[k].replace(nambu=None))
    if isinstance(operator,FOperator):
        indices,sms=[],[]
        TABLE,keys=table.reversal,list(ms.keys())
        sign=np.array([[1.0,0.0],[0.0,-1.0]],dtype=dtype)
        for leaf in range(keys[0],keys[-1]+1):
            if leaf in ms:
                assert counts[0] in (1,2)
                length-=counts.pop(0)
                indices.append(inds.pop(0))
                sms.append(SpinMatrix(0.5,tags.pop(0)+('' if length%2==0 else 's'),matrix=ms[leaf] if length%2==0 else ms[leaf].dot(sign),dtype=dtype))
            elif length%2!=0:
                indices.append(TABLE[leaf].replace(nambu=None))
                sms.append(SpinMatrix(0.5,'s',matrix=sign,dtype=dtype))
        return SOperator(value=operator.value*parity(permutation),indices=indices,spins=sms)
    else:
        sms=[SpinMatrix(0.5,tag,matrix=matrix,dtype=dtype) for tag,matrix in zip(tags,iter(ms.values()))]
        return SOperator(value=operator.value,indices=inds,spins=sms)

def twisttransformation(operator,vectors,thetas):
    '''
    Apply twisted boundary conditions to an operator.

    Parameters
    ----------
    operator : FockOperator
        The original Fock operator.
    vectors : 2d ndarray-like
        The translation vectors of the lattice.
    thetas : 1d ndarray-like
        The twisted angles.

    Returns
    -------
    FockOperator
        The new operator.
    '''
    assert isinstance(operator,FockOperator)
    assert len(vectors) in {1,2}
    if len(vectors)==1:
        assert isparallel(operator.icoord,vectors[0])
        icoord=np.array([np.linalg.norm(operator.icoord)/np.linalg.norm(vectors[0])])
    else:
        icoord=np.linalg.inv(np.asarray(vectors).T).dot(operator.icoord)
        assert np.max(np.abs(icoord-np.around(icoord)))<RZERO
    phase,value=np.exp(2.0j*np.pi*icoord.dot(thetas)),operator.value
    if operator.rank in {1,2}:
        if operator.indices[0].nambu==CREATION:
            value*=np.conjugate(phase)
        else:
            value*=phase
    elif operator.rank==4:
        index1,index2=operator.indices[0:2]
        if index1.nambu==CREATION and index2.nambu==CREATION:
            value*=np.conjugate(phase)**2
        elif index1.nambu==ANNIHILATION and index2.nambu==ANNIHILATION:
            value*=phase**2
    else:
        raise ValueError('twisttransformation error: not supported rank(%s).'%operator.rank)
    result=FockOperator.__new__(operator.__class__)
    super(FockOperator,result).__init__(value)
    result.indices=operator.indices
    result.seqs=operator.seqs
    result.rcoord=operator.rcoord
    result.icoord=operator.icoord
    return result