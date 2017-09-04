'''
====================
Matrix product state
====================

Matrix product states, including:
    * classes: MPS, Vidal
'''

__all__=['MPS','Vidal']

import numpy as np
from numpy.linalg import norm
from HamiltonianPy import QuantumNumber as QN
from HamiltonianPy import QuantumNumbers as QNS
from ..Misc import TOL,Arithmetic
from Tensor import *
from copy import copy,deepcopy

class MPS(Arithmetic,list):
    '''
    The general matrix product state, with each of its elements being a 3d `Tensor`.

    Attributes
    ----------
    mode : 'NB' or 'QN'
        'NB' for not using good quantum number;
        'QN' for using good quantum number.
    Lambda : Tensor
        The Lambda matrix (singular values) on the connecting link.
    cut : integer
        The index of the connecting link.

    Notes
    -----
    The left-canonical MPS, right-canonical MPS and mixed-canonical MPS are considered as special cases of this form.
    '''
    L,S,R=0,1,2

    def __init__(self,mode='NB',ms=(),Lambda=None,cut=None,sites=None,bonds=None):
        '''
        Constructor.

        Parameters
        ----------
        mode : 'NB' or 'QN', optional
            'NB' for not using good quantum number;
            'QN' for using good quantum number.
        ms : list of 3d Tensor/ndarray, optional
            The matrices.
        Lambda : 1d ndarray/Tensor, optional
            The Lambda matrix (singular values) on the connecting bond.
        cut : integer, optional
            The position of the connecting bond.
        sites : list of Label, optional
            The labels for the physical legs.
        bonds : list of Label, optional
            The labels for the virtual legs.
        '''
        assert mode in ('QN','NB') and (Lambda is None)==(cut is None) and (sites is None)==(bonds is None)
        self.mode=mode
        if Lambda is None and cut is None:
            self.Lambda=None
            self.cut=None
        else:
            assert 0<cut<=len(ms)
        if sites is None:
            for i,m in enumerate(ms):
                assert isinstance(m,Tensor) and m.ndim==3
                self.append(m)
            if Lambda is not None and cut is not None:
                assert isinstance(Lambda,Tensor)
                self.Lambda=Lambda
                self.cut=cut
        else:
            assert len(ms)==len(sites) and len(ms)==len(bonds)-1
            for i in xrange(len(ms)):
                assert ms[i].ndim==3
                self.append(Tensor(ms[i],labels=[bonds[i],sites[i],bonds[i+1]]))
            if Lambda is not None and cut is not None:
                self.Lambda=Tensor(Lambda,labels=deepcopy([bonds[cut]]))
                self.cut=cut

    @staticmethod
    def from_state(state,sites,bonds,cut=0,nmax=None,tol=None):
        '''
        Convert the normal representation of a state to the matrix product representation.

        Parameters
        ----------
        state : 1d ndarray
            The normal representation of a state.
        sites : list of Label
            The labels for the physical legs.
        bonds : list of Label
            The labels for the virtual legs.
        cut : integer, optional
            The index of the connecting link.
        nmax : integer, optional
            The maximum number of singular values to be kept.
        tol : float64, optional
            The tolerance of the singular values.

        Returns
        -------
        MPS
            The corresponding mixed-canonical mps.
        '''
        assert state.ndim==1 and len(sites)+1==len(bonds)
        L,R=Label('__MPS_from_state_L__',qns=bonds[+0].qns),Label('__MPS_from_state_R__',qns=bonds[-1].qns)
        S=Label('__MPS_from_state_S__',qns=QNS.kron([label.qns for label in sites]) if L.qnon else np.product([label.dim for label in sites]))
        m=Tensor(state.reshape((1,-1,1)),labels=[L,S,R])
        if cut==0:
            u,s,ms=m.expanded_svd(L=[L],S=S,R=[R],E=sites,I=bonds[:-1],nmax=nmax,tol=tol,cut=0)
            Lambda=Tensor(np.asarray(contract([u,s],engine='tensordot')).reshape((-1,)),labels=[bonds[0]])
            ms[-1].relabel(olds=[R],news=[bonds[-1]])
        elif cut==len(sites):
            ms,s,v=m.expanded_svd(L=[L],S=S,R=[R],E=sites,I=bonds[1:],nmax=nmax,tol=tol,cut=len(sites))
            Lambda=Tensor(np.asarray(contract([s,v],engine='tensordot')).reshape((-1,)),labels=[bonds[-1]])
            ms[0].relabel(olds=[L],news=[bonds[0]])
        else:
            ms,Lambda=m.expanded_svd(L=[L],S=S,R=[R],E=sites,I=bonds[1:-1],nmax=nmax,tol=tol,cut=cut)
            ms[+0].relabel(olds=[L],news=[bonds[+0]])
            ms[-1].relabel(olds=[R],news=[bonds[-1]])
        return MPS(mode='QN' if L.qnon else 'NB',ms=ms,Lambda=Lambda,cut=cut)

    @staticmethod
    def productstate(ms,sites,bonds):
        '''
        Generate a product state.

        Parameters
        ----------
        ms : list of 1d ndarray
            The matrices of the product state. 
        sites : list of Label
            The site labels of the product state.
        bonds : list of Label
            The bond labels of the product state.

        Returns
        -------
        MPS
            The corresponding product state.
        '''
        assert len(ms)==len(sites) and len(bonds)==len(sites)+1
        for i in xrange(len(ms)):
            m,L,S,R=ms[i],bonds[i],sites[i],bonds[i+1]
            ms[i]=Tensor(m.reshape((1,S.dim,1)),labels=[L,S,R])
            if S.qnon: ms[i].qng(axes=[0,1],qnses=[L.qns,S.qns],signs='++')
        return MPS(mode='QN' if S.qnon else 'NB',ms=ms)

    @property
    def state(self):
        '''
        Convert to the normal representation.

        Returns
        -------
        result : ndarray
            * When `self` is a pure state, `result` is 1d and the norm of the mps is omitted.
            * When `self` is a mixed state, `result` is 2d with the rows being the pure states and the singular value for each pure state is omitted.
        '''
        table=self.table
        if self.cut is None:
            result=contract(self,engine='tensordot')
        elif self.cut==0:
            result=contract([self.Lambda]+list(self),engine='tensordot')
        elif self.cut==self.nsite:
            result=contract(list(self)+[self.Lambda],engine='tensordot')
        else:
            A=contract(self.As,engine='tensordot')
            B=contract(self.Bs,engine='tensordot')
            result=contract([A,self.Lambda,B],engine='einsum')
        L,R=self[0].labels[MPS.L],self[-1].labels[MPS.R]
        if L==R:
            return np.asarray(result).reshape((-1,))
        else:
            assert L.dim==1 or R.dim==1
            if L.dim==1 and R.dim==1:
                return np.asarray(result).reshape((-1,))
            elif L.dim==1:
                return np.asarray(result).reshape((-1,R.dim)).T
            else:
                return np.asarray(result).reshape((L.dim,-1))

    @staticmethod
    def random(sites,bonds,cut=None,nmax=None,dtype=np.float64):
        '''
        Generate a random mps.

        Parameters
        ----------
        sites : list of Label/integer/QuantumNumbers
            The labels/number-of-degrees-of-freedom/quantum-numbers of the physical legs.
        bonds :
            * list of Label
                The labels of the virtual legs.
            * 2-list of QuantumNumber 
                The quantum numbers of the first and last virtual legs.
            * QuantumNumber
                The quantum numbers of the last virtual legs.
        cut : integer, optional
            The index of the connecting link.
        nmax : integer, optional
            The maximum number of singular values to be kept.
        dtype : np.float32, np.float64, np.complex64, np.complex128, optional
            The data type of the random mps.

        Returns
        -------
        MPS
            The random mixed-canonical mps.
        '''
        np.random.seed()
        sites=[site if isinstance(site,Label) else Label('__MPS_RANDOM_S_%s__'%i,qns=site) for i,site in enumerate(sites)]
        if all(isinstance(bond,Label) for bond in bonds):
            assert len(bonds)==len(sites)+1
        else:
            if isinstance(bonds,QN):
                iqns,oqns=QNS.mono(bonds[0].zero()),QNS.mono(bonds[0])
            else:
                assert len(bonds)==2
                iqns=QNS.mono(bonds[0]) if isinstance(bonds[0],QN) else bonds[0]
                oqns=QNS.mono(bonds[1]) if isinstance(bonds[1],QN) else bonds[0]
            bonds=[Label('__MPS_RANDOM_B_%s__'%i,qns=iqns if i==0 else (oqns if i==len(sites) else None)) for i in xrange(len(sites)+1)]
        mode,shape='QN' if next(iter(sites)).qnon else 'NB',tuple([site.dim for site in sites])
        if mode=='QN':
            result=0
            if dtype in (np.float32,np.float64):
                coeffs=np.random.random(nmax)
            else:
                coeffs=np.random.random(nmax)+1j*np.random.random(nmax)
            for k,indices in enumerate(QNS.decomposition([site.qns for site in sites],bonds[-1].qns[0]-bonds[0].qns[0],method='monte carlo',nmax=nmax)):
                ms=[np.array([1.0 if i==index else 0.0 for i in xrange(site.dim)],dtype=dtype) for site,index in zip(sites,indices)]
                result+=MPS.productstate(ms,sites,deepcopy(bonds))*coeffs[k]
        else:
            ms=[]
            for i in xrange(len(sites)):
                if dtype in (np.float32,np.float64):
                    ms.append(np.random.random((nmax,shape[i],nmax)))
                else:
                    ms.append(np.random.random((nmax,shape[i],nmax))+1j*np.random.random((nmax,shape[i],nmax)))
            result=MPS(mode=mode,ms=ms,sites=sites,bonds=bonds)
        if cut is None:
            result.canonicalize(cut=len(sites)/2,nmax=nmax)
            result._merge_ABL_()
        else:
            result.canonicalize(cut=cut,nmax=nmax)
        return result

    @staticmethod
    def concatenate(mpses,mode=None,cut=None):
        '''
        Concatenate several mpses into one.

        Parameters
        ----------
        mpses : list of MPS
            The mpses to be concatenated.
        mode : 'QN' or 'NB', optional
            The mode of the result. Only when ``len(mpses)==0`` will it be considered.
        cut : integer, optional
            The position of the connecting bond after the canonicalization.

        Returns
        -------
        MPS
            The result.
        '''
        modes=np.array([mps.mode=='QN' for mps in mpses])
        assert all(modes) or all(~modes)
        if len(mpses)==0:
            result=MPS(mode=mode)
        else:
            result=copy(mpses[0])
            result.reset()
            for mps in mpses[1:]:
                assert result[-1].labels[MPS.R]==mps[0].labels[MPS.L]
                m,Lambda=mps._merge_ABL_()
                result.extend(mps)
                mps._set_ABL_(m,Lambda)
            if cut is not None: result.canonicalize(cut=cut)
        return result

    @property
    def nmax(self):
        '''
        The maximum bond dimension of the mps.
        '''
        return max(bond.dim for bond in self.bonds) if self.nsite>0 else None

    @property
    def table(self):
        '''
        The table of the mps.

        Returns
        -------
        dict
            For each of its (key,value) pair,
                * key: Label
                    The site label of each matrix in the mps.
                * value: integer
                    The index of the corresponding matrix in the mps.
        '''
        return {m.labels[MPS.S]:i for i,m in enumerate(self)}

    @property
    def As(self):
        '''
        The A matrices.
        '''
        return self[0:self.cut]

    @property
    def Bs(self):
        '''
        The B matrices.
        '''
        return self[self.cut:self.nsite]

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        result=['L: %s\nS: %s\nR: %s\ndata:\n%s'%(m.labels[0],m.labels[1],m.labels[2],np.asarray(m)) for m in self]
        if self.Lambda is not None:
            result.insert(self.cut,'Lambda: %s\ndata:\n%s'%(self.Lambda.labels[0],np.asarray(self.Lambda)))
        return '\n'.join(result)

    @property
    def nsite(self):
        '''
        The number of total sites.
        '''
        return len(self)

    @property
    def norm(self):
        '''
        The norm of the matrix product state.
        '''
        temp=copy(self)
        temp.reset(cut=0)
        temp>>=temp.nsite
        return np.asarray(temp.Lambda)

    @property
    def sites(self):
        '''
        The site labels of the mps.
        '''
        return [m.labels[MPS.S] for m in self]

    @property
    def bonds(self):
        '''
        The bond labels of the mps.
        '''
        result=[]
        for i,m in enumerate(self):
            if i==0: result.append(m.labels[MPS.L])
            result.append(m.labels[MPS.R])
        return result

    def relabel(self,sites,bonds):
        '''
        Change the labels of the mps.

        Parameters
        ----------
        sites : list of Label
            The new site labels of the mps.
        bonds : list of Label
            The new bond labels of the mps.
        '''
        assert len(sites)==self.nsite and len(bonds)==self.nsite+1
        for m,L,S,R in zip(self,bonds[:-1],sites,bonds[1:]):
            m.relabel(news=[L,S,R])
        if self.Lambda is not None:
            self.Lambda.relabel(news=[bonds[self.cut]])

    def copy(self,copy_data=False):
        '''
        Make a copy of the mps.

        Parameters
        ----------
        copy_data : logical, optional
            * When True, both the labels and data of each tensor in this mps will be copied;
            * When False, only the labels of each tensor in this mps will be copied.

        Returns
        -------
        MPS
            The copy of self.
        '''
        ms=[m.copy(copy_data=copy_data) for m in self]
        Lambda=None if self.Lambda is None else self.Lambda.copy(copy_data=copy_data)
        return MPS(ms=ms,Lambda=Lambda,cut=self.cut)

    def is_canonical(self):
        '''
        Judge whether each site of the MPS is in the canonical form.
        '''
        result=[]
        for i,M in enumerate(self):
            temp=[np.asarray(M.take(indices=j,axis=self.S)) for j in xrange(M.shape[self.S])]
            buff=None
            for matrix in temp:
                if buff is None:
                    buff=matrix.T.conjugate().dot(matrix) if i<self.cut else matrix.dot(matrix.T.conjugate())
                else:
                    buff+=matrix.T.conjugate().dot(matrix) if i<self.cut else matrix.dot(matrix.T.conjugate())
            result.append((abs(buff-np.identity(M.shape[MPS.R if i<self.cut else MPS.L]))<TOL).all())
        return result

    def canonicalize(self,cut=0,nmax=None,tol=None):
        '''
        Canonicalize an mps by svd.

        Parameters
        ----------
        cut : integer, optional
            The position of the connecting bond after the canonicalization.
        nmax : integer, optional
            The maximum number of singular values to be kept.
        tol : float64, optional
            The tolerance of the singular values.
        '''
        if cut<=self.nsite/2:
            self.reset(cut=self.nsite)
            self<<=(self.nsite,nmax,tol)
            self>>=(cut,nmax,tol)
        else:
            self.reset(cut=0)
            self>>=(self.nsite,nmax,tol)
            self<<=(self.nsite-cut,nmax,tol)

    def compress(self,nsweep=1,cut=0,nmax=None,tol=None):
        '''
        Compress an mps by svd.

        Parameters
        ----------
        nsweep : integer, optional
            The number of sweeps to compress the mps.
        cut : integer, optional
            The position of the connecting bond after the compression.
        nmax : integer, optional
            The maximum number of singular values to be kept.
        tol : float64, optional
            The tolerance of the singular values.
        '''
        for sweep in xrange(nsweep):
            self.canonicalize(cut=cut,nmax=nmax,tol=tol)

    def reset(self,cut=None):
        '''
        Reset the position of the connecting bond of the mps.

        Parameters
        ----------
        cut : None or integer, optional
            The position of the new connecting bond.
        '''
        self._merge_ABL_()
        if cut is not None:
            if 0<=cut<=self.nsite:
                self.cut=cut
                self.Lambda=Tensor(1.0,labels=[])
            else:
                raise ValueError("MPS reset error: cut(%s) should be None or in the range [%s,%s]"%(cut,0,self.nsite))

    def _merge_ABL_(self,merge='R'):
        '''
        Merge the Lambda matrix on the connecting bond to its left or right neighbouring matrix.

        Parameters
        ----------
        merge : 'L','R', optional
            * When 'L', self.Lambda will be merged into its left neighbouring matrix;
            * When 'R', self.Lambda will be merged into its right neighbouring matrix.

        Returns
        -------
        m : 3d Tensor
            The original left/right neighbouring matrix.
        Lambda : 1d Tensor
            The original Lambda matrix.

        Notes
        -----
        * When ``self.cut==0``, the `Lambda` matrix will be merged with `self[0]` no matter what merge is, and
        * When ``self.cut==self.nsite``, the `Lambda` matrix will be merged with `self[-1]` no matter what merge is.
        '''
        if self.cut is not None:
            assert merge.upper() in ('L','R')
            merge='L' if self.cut==self.nsite else ('R' if self.cut==0 else merge.upper())
            if merge=='L':
                m,Lambda=self[self.cut-1],self.Lambda
                self[self.cut-1]=contract([self[self.cut-1],self.Lambda],engine='einsum',reserve=[self[self.cut-1].labels[MPS.R]])
            else:
                m,Lambda=self[self.cut],self.Lambda
                self[self.cut]=contract([self.Lambda,self[self.cut]],engine='einsum',reserve=[self[self.cut].labels[MPS.L]])
            self.cut=None
            self.Lambda=None
        else:
            m,Lambda=None,None
        return m,Lambda

    def _set_ABL_(self,m,Lambda):
        '''
        Set the matrix at a certain position and the Lambda of an mps.

        Parameters
        ----------
        m : Tensor
            The matrix at a certain position of the mps.
        Lambda : Tensor
            The singular values at the connecting link of the mps.
        '''
        if isinstance(m,Tensor) and isinstance(Lambda,Tensor):
            assert m.ndim==3 and Lambda.ndim==1
            L,S,R=m.labels[MPS.L],m.labels[MPS.S],m.labels[MPS.R]
            pos=self.table[S]
            self[pos]=m
            self.Lambda=Lambda
            if Lambda.labels[0]==L:
                self.cut=pos
            elif Lambda.labels[0]==R:
                self.cut=pos+1
            else:
                raise ValueError("MPS _set_ABL_ error: the labels of m(%s) and Lambda(%s) do not match."%(m.labels,Lambda.labels))

    def _set_B_and_lmove_(self,M,nmax=None,tol=None):
        '''
        Set the B matrix at self.cut and move leftward.

        Parameters
        ----------
        M : Tensor
            The tensor used to set the B matrix.
        nmax : integer, optional
            The maximum number of singular values to be kept. 
        tol : float64, optional
            The truncation tolerance.
        '''
        if self.cut==0: raise ValueError('MPS _set_B_and_lmove_ error: the cut is already zero.')
        L,S,R=M.labels[MPS.L],M.labels[MPS.S],M.labels[MPS.R]
        u,s,v,err=M.svd(row=[L],new=L.prime,col=[S,R],row_signs='+',col_signs='-+',nmax=nmax,tol=tol,return_truncation_err=True)
        v.relabel(olds=s.labels,news=[L.replace(qns=s.labels[0].qns)])
        self[self.cut-1]=v
        if self.cut==1:
            self.Lambda=contract([u,s],engine='tensordot')
        else:
            self[self.cut-2]=contract([self[self.cut-2],u],engine='tensordot')
            self[self.cut-2].relabel(olds=s.labels,news=[L.replace(qns=s.labels[0].qns)])
            s.relabel(news=[L.replace(qns=s.labels[0].qns)])
            self.Lambda=s
        self.cut=self.cut-1

    def _set_A_and_rmove_(self,M,nmax=None,tol=None):
        '''
        Set the A matrix at self.cut and move rightward.

        Parameters
        ----------
        M : Tensor
            The tensor used to set the A matrix.
        nmax : integer, optional
            The maximum number of singular values to be kept. 
        tol : float64, optional
            The truncation tolerance.
        '''
        if self.cut==self.nsite:
            raise ValueError('MPS _set_A_and_rmove_ error: the cut is already maximum.')
        L,S,R=M.labels[MPS.L],M.labels[MPS.S],M.labels[MPS.R]
        u,s,v,err=M.svd(row=[L,S],new=R.prime,col=[R],row_signs='++',col_signs='+',nmax=nmax,tol=tol,return_truncation_err=True)
        u.relabel(olds=s.labels,news=[R.replace(qns=s.labels[0].qns)])
        self[self.cut]=u
        if self.cut==self.nsite-1:
            self.Lambda=contract([s,v],engine='tensordot')
        else:
            self[self.cut+1]=contract([v,self[self.cut+1]],engine='tensordot')
            self[self.cut+1].relabel(olds=s.labels,news=[R.replace(qns=s.labels[0].qns)])
            s.relabel(news=[R.replace(qns=s.labels[0].qns)])
            self.Lambda=s
        self.cut=self.cut+1

    def __ilshift__(self,other):
        '''
        Operator "<<=", which shift the connecting link leftward by a non-negative integer.

        Parameters
        ----------
        other : internal or 3-tuple
            * integer
                The number of times that `self.cut` will move leftward.
            * 3-tuple in the form (k,nmax,tol)
                * k: integer
                    The number of times that `self.cut` will move leftward.
                * nmax: integer
                    The maximum number of singular values to be kept.
                * tol: float64
                    The truncation tolerance.
        '''
        nmax,tol=None,None
        if isinstance(other,tuple):
            k,nmax,tol=other
        else:
            k=other
        if k>=0:
            for i in xrange(k):
                self._set_B_and_lmove_(contract([self[self.cut-1],self.Lambda],engine='einsum',reserve=[self[self.cut-1].labels[MPS.R]]),nmax,tol)
        else:
            for i in xrange(-k):
                self._set_A_and_rmove_(contract([self.Lambda,self[self.cut]],engine='einsum',reserve=[self[self.cut].labels[MPS.L]]),nmax,tol)
        return self

    def __lshift__(self,other):
        '''
        Operator "<<".

        Parameters
        ----------
        other : integer or 3-tuple.
            Please see MPS.__ilshift__ for details.
        '''
        return copy(self).__ilshift__(other)

    def __irshift__(self,other):
        '''
        Operator ">>=", which shift the connecting link rightward by a non-negative integer.

        Parameters
        ----------
        other: integer or 3-tuple
            * integer
                The number of times that self.cut will move rightward.
            * 3-tuple in the form (k,nmax,tol)
                * k: integer
                    The number of times that self.cut will move rightward.
                * nmax: integer
                    The maximum number of singular values to be kept.
                * tol: float64
                    The truncation tolerance.
        '''
        return self.__ilshift__((-other[0],other[1],other[2]) if isinstance(other,tuple) else -other)

    def __rshift__(self,other):
        '''
        Operator ">>".

        Parameters
        ----------
        other : integer or 3-tuple.
            Please see MPS.__irshift__ for details.
        '''
        return copy(self).__irshift__(other)

    def __add__(self,other):
        '''
        Overloaded addition(+) operator, which supports the addition of two mpses.
        '''
        if isinstance(other,MPS):
            assert self.mode==other.mode and self.nsite==other.nsite
            if self is other:
                u,Lambda=self._merge_ABL_()
            else:
                u1,Lambda1=self._merge_ABL_()
                u2,Lambda2=other._merge_ABL_()
            mode,ms=self.mode,[]
            for i,(m1,m2) in enumerate(zip(self,other)):
                assert m1.labels==m2.labels
                labels=[label.replace(qns=None) for label in m1.labels]
                axes=[MPS.L,MPS.S] if i==0 else ([MPS.S,MPS.R] if i==self.nsite-1 else [MPS.S])
                ms.append(Tensor.directsum([m1,m2],labels=labels,axes=axes))
            if self is other:
                self._set_ABL_(u,Lambda)
            else:
                self._set_ABL_(u1,Lambda1)
                other._set_ABL_(u2,Lambda2)
            return MPS(mode=mode,ms=ms)
        else:
            assert norm(other)==0
            return self

    __iadd__=__add__

    def __imul__(self,other):
        '''
        Overloaded self-multiplication(*=) operator, which supports the self-multiplication by a scalar.
        '''
        if self.Lambda is None:
            self[0]*=other
        else:
            self.Lambda*=other
        return self

    def __mul__(self,other):
        '''
        Overloaded multiplication(*) operator, which supports the multiplication of an mps with a scalar.
        '''
        result=copy(self)
        if result.Lambda is None:
            result[0]=result[0]*other
        else:
            result.Lambda=result.Lambda*other
        return result

    @staticmethod
    def overlap(mps1,mps2):
        '''
        The overlap between two mps.

        Parameters
        ----------
        mps1,mps2 : MPS
            The MPS between which the overlap is calculated.

        Returns
        -------
        number
            The overlap.
        '''
        if mps1 is mps2:
            u,Lambda=mps1._merge_ABL_()
        else:
            u1,Lambda1=mps1._merge_ABL_()
            u2,Lambda2=mps2._merge_ABL_()
        result=[]
        for i,(m1,m2) in enumerate(zip(mps1,mps2)):
            assert m1.labels==m2.labels
            L,R=m1.labels[MPS.L],m1.labels[MPS.R]
            news,olds=[L.prime,R.prime],[L,R]
            m1=m1.copy(copy_data=False).conjugate()
            if i==0:
                news.remove(L.prime)
                olds.remove(L)
            if i==mps1.nsite-1:
                news.remove(R.prime)
                olds.remove(R)
            m1.relabel(news=news,olds=olds)
            result.append(m1)
            result.append(m2)
        if mps1 is mps2:
            mps1._set_ABL_(u,Lambda)
        else:
            mps1._set_ABL_(u1,Lambda1)
            mps2._set_ABL_(u2,Lambda2)
        return np.asarray(contract(result,engine='tensordot'))

    def relayer(self,degfres,layer,nmax=None,tol=None):
        '''
        Construct a new mps with the physical indices confined on a specific layer of degfres.

        Parameters
        ----------
        degfres : DegFreTree
            The tree of the physical degrees of freedom.
        layer : integer/tuple-of-string
            The layer where the physical indices are confined.
        nmax : integer, optional
            The maximum number of singular values to be kept.
        tol : np.float64, optional
            The tolerance of the singular values.

        Returns
        -------
        MPS
            The new mps.
        '''
        new=layer if type(layer) in (int,long) else degfres.layers.index(layer)
        assert 0<=new<len(degfres.layers)
        old=degfres.level(next(iter(self)).labels[MPS.S].identifier)-1
        if new==old:
            return copy(self)
        else:
            t,svs=self._merge_ABL_()
            olayer,nlayer=degfres.layers[old],degfres.layers[new]
            sites,bonds=degfres.labels('S',nlayer),degfres.labels('B',nlayer)
            Ms=[]
            if new<old:
                table=degfres.table(olayer)
                for i,site in enumerate(sites):
                    M=contract([self[table[index]] for index in degfres.descendants(site.identifier,generation=old-new)],engine='tensordot')
                    o1,o2=M.labels[0],M.labels[-1]
                    n1,n2=bonds[i].replace(qns=o1.qns),bonds[i+1].replace(qns=o2.qns)
                    M.relabel(olds=[o1,o2],news=[n1,n2])
                    Ms.append(M.merge((M.labels[1:-1],site)))
                Lambda,cut=None,None
            else:
                table=degfres.table(nlayer)
                for i,m in enumerate(self):
                    if i>0: m=contract([s,v,m],engine='einsum',reserve=s.labels)
                    L,S,R=m.labels
                    indices=degfres.descendants(S.identifier,generation=new-old)
                    start,stop=table[indices[0]],table[indices[-1]]+1
                    us,s,v=m.expanded_svd(L=[L],S=S,R=[R],E=sites[start:stop],I=bonds[start+1:stop+1],cut=stop-start,nmax=nmax,tol=tol)
                    Ms.extend(us)
                Lambda,cut=contract([s,v],engine='tensordot'),len(Ms)
                Ms[0].relabel(olds=[MPS.L],news=[bonds[0].replace(qns=Ms[0].labels[MPS.L].qns)])
                Lambda.relabel([bonds[-1].replace(qns=Lambda.labels[0].qns)])
            self._set_ABL_(t,svs)
            return MPS(mode=self.mode,ms=Ms,Lambda=Lambda,cut=cut)

class Vidal(object):
    '''
    The Vidal canonical matrix product state.

    Attributes
    ----------
    Gammas : list of Tensor
        The Gamma matrices on the site.
    Lambdas : list of Tensor
        The Lambda matrices (singular values) on the link.
    '''
    L,S,R=0,1,2

    def __init__(self,Gammas,Lambdas,sites=None,bonds=None):
        '''
        Constructor.

        Parameters
        ----------
        Gammas : list of 3d ndarray/Tensor
            The Gamma matrices on the site.
        Lambdas : list of 1d ndarray/Tensor
            The Lambda matrices (singular values) on the link.
        sites : list of Label, optional
            The labels for the physical legs.
        bonds : list of Label, optional
            The labels for the virtual legs.
        '''
        assert len(Gammas)==len(Lambdas)+1 and (sites is None)==(bonds is None)
        self.Gammas=[]
        self.Lambdas=[]
        if sites is None:
            for Gamma in Gammas:
                assert isinstance(Gamma,Tensor)
                assert Gamma.ndim==3
                self.Gammas.append(Gamma)
            for Lambda in Lambdas:
                assert isinstance(Lambda,Tensor)
                assert Lambda.ndim==1
                self.Lambdas.append(Lambda)
        else:
            assert len(Gammas)==len(sites) and len(Gammas)==len(bonds)-1
            for Gamma,L,S,R in zip(Gammas,bonds[:-1],sites,bonds[1:]):
                assert Gamma.ndim==3
                self.Gammas.append(Tensor(Gamma,labels=[L,S,R]))
            for Lambda,label in zip(Lambdas,bonds[1:-1]):
                assert Lambda.ndim==1
                self.Lambdas.append(Tensor(Lambda,labels=[label]))

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        for i,Gamma in enumerate(self.Gammas):
            result.append(str(Gamma))
            if i<len(self.Gammas)-1:
                result.append(str(self.Lambdas[i]))
        return '\n'.join(result)

    @property
    def nsite(self):
        '''
        The number of total sites.
        '''
        return len(self.Gammas)

    @property
    def state(self):
        '''
        Convert to the normal representation.

        Returns
        -------
        1d ndarray
            The corresponding normal representation of the state.
        '''
        result=None
        for i,Gamma in enumerate(self.Gammas):
            if result is None:
                result=Gamma
            else:
                result=contract([result,self.Lambdas[i-1],Gamma],engine='einsum')
        return np.asarray(result).reshape((-1,))

    def to_mixed(self,cut):
        '''
        Convert to the mixed MPS representation.

        Parameters
        ----------
        cut : integer
            The index of the connecting link.

        Returns
        -------
        MPS
            The corresponding mixed MPS.
        '''
        ms,Lambda=[],None
        for i,Gamma in enumerate(self.Gammas):
            if i>0 and i==cut: Lambda=self.Lambdas[i-1]
            if i<cut:
                ms.append(Gamma if i==0 else contract([self.Lambdas[i-1],Gamma],engine='einsum',reserve=self.Lambdas[i-1].labels))
            else:
                ms.append(contract([Gamma,self.Lambdas[i]],engine='einsum',reserve=self.Lambdas[i].labels) if i<self.nsite-1 else Gamma)
        return MPS(ms=ms,Lambda=Lambda,cut=cut)
