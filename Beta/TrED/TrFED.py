'''
1D fermionic exact diagonalization with translation symmetry, including:
    * classes: TRBasis, TrFED, EB
    * functions: trfoptrep, TrFEDEB
'''

__all__=['TRBasis','trfoptrep','TrFED','EB','TrFEDEB']

from fbasis import *
from numba import jit
from math import sqrt
from scipy.sparse import csr_matrix
import numpy as np
import HamiltonianPy as HP
import HamiltonianPy.ED as ED
import HamiltonianPy.Misc as HM
import matplotlib.pyplot as plt

class TRBasis(HP.FBasis):
    def __init__(self,basis,dk,nk):
        assert isinstance(basis,HP.FBasis) and basis.mode in ('FP','FS')
        self.dk=dk
        self.nk=nk
        self.mode='%sTR'%basis.mode
        self.nstate=basis.nstate
        self.nparticle=basis.nparticle
        self.table=basis.table
        seqs,maps,translations,signs,nbasis=trbasis(self.table,dk,nk,self.nstate.sum())
        self.seqs=seqs[:nbasis]
        self.maps=maps
        self.translations=translations
        self.signs=signs
        self.masks=trmasks(self.seqs,self.translations,self.signs,nk)

    def tostr(self,protocol=0):
        assert protocol in (0,1)
        if protocol==0:
            return '\n'.join('{}({}): {:b}'.format(i,self.translations[index],self.table[index]) for i,index in enumerate(self.seqs))
        else:
            return '\n'.join('{:b}({},{}): {:b}'.format(basis,translation,sign,self.table[self.seqs[index]]) for basis,index,translation,sign in zip(self.table,self.maps,self.translations,self.signs))

    @property
    def nbasis(self):
        return len(self.seqs)

    def indices(self,k):
        return np.concatenate(np.argwhere(self.masks[k,:]>=0))

def trfoptrep(operator,k,basis,dtype=np.complex128):
    assert operator.rank%2==0 and isinstance(basis,TRBasis) and 0<=k<basis.nk
    value,nambus,sequences=operator.value,(np.array([index.nambu for index in operator.indices])>0)[::-1],np.array(operator.seqs)[::-1]
    table,seqs,maps,translations,signs,masks,nk=basis.table,basis.seqs,basis.maps,basis.translations,basis.signs,basis.masks,basis.nk
    data,indices,indptr,dim=_trfoptrep_(k,value,nambus,sequences,table,seqs,maps,translations,signs,masks,nk,dtype)
    return csr_matrix((data,indices,indptr),shape=(dim,dim)).T

@jit
def _trfoptrep_(k,value,nambus,sequences,table,seqs,maps,translations,signs,masks,nk,dtype):
    ndata,data,indices,indptr=0,np.zeros(len(seqs),dtype=dtype),np.zeros(len(seqs),dtype=np.int32),np.zeros(len(seqs)+1,dtype=np.int32)
    dim,eye,temp=0,long(1),np.zeros(len(sequences)+1,dtype=np.int64)
    factor=np.exp(1.0j*2*np.pi*k/nk)
    for i in xrange(len(seqs)):
        if masks[k,i]>=0:
            indptr[dim]=ndata
            temp[0]=table[seqs[i]]
            for m in xrange(len(sequences)):
                if bool(temp[m]&eye<<sequences[m])==nambus[m]: break
                temp[m+1]=temp[m]|eye<<sequences[m] if nambus[m] else temp[m]&~(eye<<sequences[m])
            else:
                seq=HP.sequence(temp[-1],table)
                j=maps[seq]
                if masks[k,j]>=0:
                    nsign=0
                    for m in xrange(len(sequences)):
                        for n in xrange(sequences[m]):
                            if temp[m]&eye<<n: nsign+=1
                    indices[ndata]=masks[k,j]
                    sign=(1 if j==i else signs[seq])*(-1)**nsign
                    phase=1 if j==i else factor**translations[seq]
                    data[ndata]=sign*value*phase*sqrt(1.0*translations[seqs[i]]/translations[seqs[j]])
                    ndata+=1
            dim+=1
    indptr[dim]=ndata
    return data[:ndata],indices[:ndata],indptr[:dim+1],dim

class TrFED(ED.FED):
    def set_matrix(self,k):
        self.matrix=0
        for operator in self.operators.itervalues():
            self.matrix+=trfoptrep(operator,k,self.basis,dtype=self.dtype)
            self.matrix+=trfoptrep(operator.dagger,k,self.basis,dtype=self.dtype)

class EB(HP.EB):
    def __init__(self,ns=6,kend=True,**karg):
        super(EB,self).__init__(**karg)
        self.ns=ns
        self.kend=kend

def TrFEDEB(engine,app):
    result=np.zeros((engine.basis.nk+(1 if app.kend else 0),app.ns+1))
    for i in xrange(engine.basis.nk):
        engine.set_matrix(i)
        result[i,0]=i
        result[i,1:]=HM.eigsh(engine.matrix,return_eigenvectors=False,which='SA',k=app.ns)
    if app.kend:
        result[-1,0]=engine.basis.nk
        result[-1,1:]=result[0,1:]
    app.result=result
    name='%s_%s'%(engine,app.name)
    if app.savedata: np.savetxt('%s/%s.dat'%(engine.dout,name),result)
    if app.plot: app.figure('L',result,'%s/%s'%(engine.dout,name))
