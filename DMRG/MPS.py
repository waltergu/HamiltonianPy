'''
Matrix product state, including:
1) classes: MPS, Vidal
'''

__all__=['MPS','Vidal']

import numpy as np
import pickle as pk
from HamiltonianPy.Basics import Label,Status,QuantumNumberCollection,bond_qnc_generation,mb_svd,expanded_svd
from HamiltonianPy.Math.Tensor import *
from HamiltonianPy.Math.linalg import truncated_svd,TOL
from copy import copy,deepcopy
from collections import OrderedDict

class MPS(list):
    '''
    The general matrix product state.
        For each of its elements: Tensor
            The matrices of the mps.
    Attributes:
        mode: 'NB' or 'QN'
            'NB' for not using good quantum number;
            'QN' for using good quantum number.
        Lambda: Tensor
            The Lambda matrix (singular values) on the connecting link.
        cut: integer
            The index of the connecting link.
    Note the left-canonical MPS, right-canonical MPS and mixed-canonical MPS are considered as special cases of this form.
    '''
    L,S,R=0,1,2

    def __init__(self,mode='NB',ms=[],labels=None,Lambda=None,cut=None):
        '''
        Constructor.
        Parameters:
            mode: 'NB' or 'QN'
                'NB' for not using good quantum number;
                'QN' for using good quantum number.
            ms: list of 3d Tensor / 3d ndarray
                The matrices.
            labels: list of 3 tuples
                The labels of the axis of the matrices, thus its length should be equal to that of ms.
                For each label in labels, 
                    label[0],label[1],label[2]: any hashable object
                        The left link / site / right link label of the matrix.
            Lambda: 1d ndarray / 1d Tensor, optional
                The Lambda matrix (singular values) on the connecting link.
            cut: integer, optional
                The index of the connecting link.
        '''
        assert mode in ('QN','NB')
        self.mode=mode
        if (Lambda is None)!=(cut is None):
            raise ValueError('MPS construction error: cut and Lambda should be both or neither be None.')
        elif Lambda is None and cut is None:
            self.Lambda=None
            self.cut=None
        elif cut<0 or cut>len(ms):
            raise ValueError('MPS construction error: the cut(%s) is out of range [0,%s].'%(cut,len(ms)))
        if labels is None:
            for i,m in enumerate(ms):
                assert isinstance(m,Tensor) and m.ndim==3
                self.append(m)
            if Lambda is not None and cut is not None:
                assert isinstance(Lambda,Tensor)
                self.Lambda=Lambda
                self.cut=cut
        else:
            assert len(ms)==len(labels)
            for m,label in zip(ms,labels):
                assert m.ndim==3
                self.append(Tensor(m,labels=list(label)))
            if Lambda is not None and cut is not None:
                if cut==0:
                    self.Lambda=Tensor(Lambda,labels=[deepcopy(labels[cut][0])])
                else:
                    self.Lambda=Tensor(Lambda,labels=[deepcopy(labels[cut-1][2])])
                self.cut=cut

    @staticmethod
    def from_state(state,shapes,labels,cut=0,nmax=None,tol=None):
        '''
        Convert the normal representation of a state to the matrix product representation.
        Parameters:
            state: 1d ndarray
                The normal representation of a state.
            shapes: list of integers
                The physical dimension of every site.
            labels: list of 3-tuple
                Please see MPS.__init__ for details.
            cut: integer, optional
                The index of the connecting link.
            namx,tol: optional
                For details, please refer to HamiltonianPy.Math.linalg.truncated_svd.
        Returns: MPS
            The corresponding mixed-canonical mps.
        '''
        if len(state.shape)!=1:
            raise ValueError("MPS.from_state error: the original state must be a pure state.")
        ms,nd=[None]*len(shapes),1
        for i in xrange(cut):
            u,s,v,err=truncated_svd(state.reshape((nd*shapes[i],-1)),full_matrices=False,nmax=nmax,tol=tol,return_truncation_err=True)
            if err>TOL: print 'MPS.from_state truncation err: %s.'%err
            ms[i]=u.reshape((nd,shapes[i],-1))
            if i==cut-1:
                if cut==len(shapes):
                    Lambda=v.transpose().dot(s)
                else:
                    Lambda,state=s,v
            else:
                state=np.einsum('i,ij->ij',s,v)
            nd=len(s)
        nd=1
        for i in xrange(len(shapes)-1,cut-1,-1):
            if i==cut:
                if cut==0:
                    u,s,v,err=truncated_svd(state.reshape((-1,shapes[i]*nd)),full_matrices=False,nmax=nmax,tol=tol,return_truncation_err=True)
                    if err>TOL:print 'MPS.from_state truncation err: %s.'%err
                    ms[i]=v.reshape((-1,shapes[i],nd))
                    Lambda=u.dot(s)
                else:
                    ms[i]=state.reshape((-1,shapes[i],nd))
            else:
                u,s,v,err=truncated_svd(state.reshape((-1,shapes[i]*nd)),full_matrices=False,nmax=nmax,tol=tol,return_truncation_err=True)
                if err>TOL:print 'MPS.from_state truncation err: %s.'%err
                ms[i]=v.reshape((-1,shapes[i],nd))
                state=np.einsum('ij,j->ij',u,s)
            nd=len(s)
        return MPS(ms=ms,labels=labels,Lambda=Lambda,cut=cut)

    def qnc_generation(self,inbond,sites):
        '''
        Generate the qnc of the MPS.
        Parameters:
            inbond: QuantumNumberCollection
                The qnc of the leftmost bond.
            sites: list of QuantumNumberCollection
                The qnc of each site index.
        '''
        assert self.mode=='NB'
        assert len(sites)==self.nsite
        old=inbond
        if self.cut==0:
            self.Lambda=old.reorder(self.Lambda)
            self.Lambda.labels[0].qnc=old
        for i,(site,m) in enumerate(zip(sites,self)):
            m=old.reorder(m,axes=[MPS.L])
            m=site.reorder(m,axes=[MPS.S])
            m.labels[MPS.L].qnc=old
            m.labels[MPS.S].qnc=site
            new=bond_qnc_generation(m,bond=old,site=site,mode='R',history=True)
            m=new.reorder(m,axes=[MPS.R])
            m.labels[MPS.R].qnc=new
            if i+1==self.cut:
                self.Lambda=new.reorder(self.Lambda)
                self.Lambda.labels[0].qnc=new
            QuantumNumberCollection.clear_history(old)
            old=new
        QuantumNumberCollection.clear_history(new)
        self.mode='QN'

    @property
    def status(self):
        '''
        The status of the MPS.
        '''
        result=OrderedDict()
        result['nsite']=self.nsite
        result['nmax']=np.array([m.shape[MPS.L] for m in self]).max() if self.nsite>0 else None
        return Status(alter=result)

    @property
    def table(self):
        '''
        The table of the mps.
        Returns: dict
            For each of its (key,value) pair,
                key: Label
                    The site label of each matrix in the mps.
                value: integer
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
        if self.cut is not None:
            result.insert(self.cut,'Lambda: %s\ndata:\n%s'%(self.Lambda.labels[0],np.asarray(self.Lambda)))
        return '\n'.join(result)

    @property
    def nsite(self):
        '''
        The number of total sites.
        '''
        return len(self)

    @property
    def state(self):
        '''
        Convert to the normal representation.
        Returns: two cases,
            1) 1d ndarray
                The MPS is a pure state.
                Its norm is omitted.
            2) 2d ndarray 
                The MPS is a mixed state with the columns being the contained pure states.
                The singular value for each pure state is omitted.
        '''
        table=self.table
        if self.cut in (0,self.nsite,None):
            result=contract(*self,sequence='sequential')
        else:
            A,B=contract(*self.As,sequence='sequential'),contract(*self.Bs,sequence='sequential')
            result=contract(A,self.Lambda,B)
        legs=set(result.labels)-set(table)
        if len(legs)==0:
            return np.asarray(result).ravel()
        elif len(legs)==2:
            if self[0].shape[MPS.L]>1:
                flag=True
            if self[-1].shape[MPS.R]>1:
                flag=False
            buff,temp=1,1
            for label,n in zip(result.labels,result.shape):
                if label not in table and n>1:
                    temp=n
                else:
                    buff*=n
            if flag:
                return np.asarray(result).reshape((temp,buff)).T
            else:
                return np.asarray(result).reshape((buff,temp))
        else:
            raise ValueError('MPS state error: %s link labels%s are left.'%(len(legs),tuple(legs)))

    @property
    def norm(self):
        '''
        The norm of the matrix product state.
        '''
        temp=copy(self)
        temp._reset_(reset=0)
        temp>>=temp.nsite
        return np.asarray(temp.Lambda)

    def _reset_(self,merge='L',reset=None):
        '''
        Reset the mps.
        This function does two things,
        1) merge the Lamdbda matrix on the link to its left neighbouring matrix or right neighbouring matrix acoording to the parameter merge,
        2) reset Lambda and cut acoording to the parameter reset.
        Parameters:
            merge: 'L' or 'R', optional
                When 'L', self.Lambda will be merged into its left neighbouring matrix;
                When 'R', self.Lambda will be merged into its right neighbouring matrix.
            reset: None or an integer, optional
                When None, self.cut and self.Lambda will be reset to None;
                When an integer, self.cut will be reset to this value and self.Lambda will be reset to a scalar Tensor with the data equal to 1.0.
        NOTE: When self.cut==0, the Lambda matrix will be merged with self[0] no matter what merge is, and
              When self.cut==self.nsite, the Lambda matrix will be merged with self[-1] no matter what merge is.
        '''
        if self.cut is not None:
            if merge=='L':
                if self.cut==0:
                    self[0]=contract(self[0],self.Lambda,reserve=[self[0].labels[MPS.L]])
                else:
                    self[self.cut-1]=contract(self[self.cut-1],self.Lambda,reserve=[self[self.cut-1].labels[MPS.R]])
            elif merge=='R':
                if self.cut==self.nsite:
                    self[-1]=contract(self[-1],self.Lambda,reserve=[self[-1].labels[MPS.R]])
                else:
                    self[self.cut]=contract(self.Lambda,self[self.cut],reserve=[self[self.cut].labels[MPS.L]])
            else:
                raise ValueError("MPS _reset_ error: merge must be 'L' or 'R' but now it is %s."%(merge))
        if reset is None:
            self.cut=None
            self.Lambda=None
        elif reset>=0 and reset<=self.nsite:
            self.cut=reset
            self.Lambda=Tensor(1.0,labels=[])
        else:
            raise ValueError("MPS _reset_ error: reset(%s) should be None or in the range [%s,%s]"%(reset,0,self.nsite))

    def _set_ABL_(self,m,Lambda):
        '''
        Set the matrix at a certain position and the Lambda of an mps.
        Parameters:
            m: Tensor
                The matrix at a certain position of the mps.
            Lambda: Tensor
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
        Parameters:
            M: Tensor
                The tensor used to set the B matrix.
            nmax: integer, optional
                The maximum number of singular values to be kept. 
            tol: float64, optional
                The truncation tolerance.
        '''
        if self.cut==0: raise ValueError('MPS _set_B_and_lmove_ error: the cut is already zero.')
        L,S,R=M.labels[self.L],M.labels[self.S],M.labels[self.R]
        m=np.asarray(M).reshape((M.shape[MPS.L],-1))
        if self.mode=='QN':
            col=S.qnc.kron(R.qnc,action='-',history=True)
            u,s,v,qnc,err=mb_svd(col.reorder(m,axes=[1]),L.qnc,-col,nmax=nmax,tol=tol,return_truncation_err=True)
            v=v[:,np.argsort(col.permutation())]
            QuantumNumberCollection.clear_history(col)
        else:
            u,s,v,qnc,err=mb_svd(m,nmax=nmax,tol=tol,return_truncation_err=True)
        self[self.cut-1]=Tensor(v.reshape((-1,M.shape[MPS.S],M.shape[MPS.R])),labels=[L.replace(qnc=qnc),S,R])
        if self.cut==1:
            if len(s)>1: raise ValueError('MPS _set_B_and_lmove_ error(not supported operation): the MPS is a mixed state.')
            self.Lambda=Tensor(np.einsum('ij,j->i',u,s),labels=[L])
        else:
            self.Lambda=Tensor(s,labels=[L.replace(qnc=qnc)])
            self[self.cut-2]=Tensor(np.einsum('ijk,kl->ijl',np.asarray(self[self.cut-2]),u),labels=self[self.cut-2].labels)
            self[self.cut-2].relabel(news=[L.replace(qnc=qnc)],olds=[L])
        self.cut=self.cut-1

    def _set_A_and_rmove_(self,M,nmax=None,tol=None):
        '''
        Set the A matrix at self.cut and move rightward.
        Parameters:
            M: Tensor
                The tensor used to set the A matrix.
            nmax: integer, optional
                The maximum number of singular values to be kept. 
            tol: float64, optional
                The truncation tolerance.
        '''
        if self.cut==self.nsite:
            raise ValueError('MPS _set_A_and_rmove_ error: the cut is already maximum.')
        L,S,R=M.labels[self.L],M.labels[self.S],M.labels[self.R]
        m=np.asarray(M).reshape((-1,M.shape[MPS.R]))
        if self.mode=='QN':
            row=L.qnc.kron(S.qnc,action='+',history=True)
            u,s,v,qnc,err=mb_svd(row.reorder(m,axes=[0]),row,R.qnc,nmax=nmax,tol=tol,return_truncation_err=True)
            u=u[np.argsort(row.permutation()),:]
            QuantumNumberCollection.clear_history(row)
        else:
            u,s,v,qnc,err=mb_svd(m,nmax=nmax,tol=tol,return_truncation_err=True)
        self[self.cut]=Tensor(u.reshape((M.shape[MPS.L],M.shape[MPS.S],-1)),labels=[L,S,R.replace(qnc=qnc)])
        if self.cut==self.nsite-1:
            if len(s)>1: raise ValueError('MPS _set_A_and_rmove_ error(not supported operation): the MPS is a mixed state.')
            self.Lambda=Tensor(np.einsum('i,ij->j',s,v),labels=[R])
        else:
            self.Lambda=Tensor(s,labels=[R.replace(qnc=qnc)])
            self[self.cut+1]=Tensor(np.einsum('ij,jkl->ikl',v,np.asarray(self[self.cut+1])),labels=self[self.cut+1].labels)
            self[self.cut+1].relabel(news=[R.replace(qnc=qnc)],olds=[R])
        self.cut=self.cut+1

    def __ilshift__(self,other):
        '''
        Operator "<<=", which shift the connecting link leftward by a non-negative integer.
        Parameters:
            other: two cases,
                1) integer
                    The number of times that self.cut will move leftward.
                2) 3-tuple
                    tuple[0]: integer
                        The number of times that self.cut will move leftward.
                    tuple[1]: integer
                        The maximum number of singular values to be kept.
                    tuple[2]: float64
                        The truncation tolerance.
        '''
        nmax,tol=None,None
        if isinstance(other,tuple):
            k,nmax,tol=other
        else:
            k=other
        for i in xrange(k):
            self._set_B_and_lmove_(contract(self[self.cut-1],self.Lambda,reserve=[self[self.cut-1].labels[MPS.R]]),nmax,tol)
        return self

    def __lshift__(self,other):
        '''
        Operator "<<".
        Parameters:
            other: integer or 3-tuple.
                Please see MPS.__ilshift__ for details.
        '''
        return copy(self).__ilshift__(other)

    def __irshift__(self,other):
        '''
        Operator ">>=", which shift the connecting link rightward by a non-negative integer.
        Parameters:
            other: two cases,
                1) integer
                    The number of times that self.cut will move rightward.
                2) 3-tuple
                    tuple[0]: integer
                        The number of times that self.cut will move rightward.
                    tuple[1]: integer
                        The maximum number of singular values to be kept.
                    tuple[2]: float64
                        The truncation tolerance.
        '''
        nmax,tol=None,None
        if isinstance(other,tuple):
            k,nmax,tol=other
        else:
            k=other
        for i in xrange(k):
            self._set_A_and_rmove_(contract(self.Lambda,self[self.cut],reserve=[self[self.cut].labels[MPS.L]]),nmax,tol)
        return self

    def __rshift__(self,other):
        '''
        Operator ">>".
        Parameters:
            other: integer or 3-tuple.
                Please see MPS.__irshift__ for details.
        '''
        return copy(self).__irshift__(other)

    def canonicalization(self,cut):
        '''
        Make the MPS in the mixed canonical form.
        Parameters:
            link: integer
                The cut of the A,B part.
        Returns: MPS
            The mixed canonical MPS.
        '''
        if self.cut<=self.nsite/2:
            self._reset_(reset=self.nsite)
            self<<=self.nsite
            self>>=cut
        else:
            self._reset_(reset=0)
            self>>=self.nsite
            self<<=(self.nsite-self.cut)

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
            result.append((abs(buff-np.identity(M.shape[self.R if i<self.cut else self.L]))<TOL).all())
        return result

    def copy(self,copy_data=False):
        '''
        Make a copy of the mps.
        Parameters:
            copy_data: logical, optional
                When True, both the labels and data of each tensor in this mps will be copied;
                When False, only the labels of each tensor in this mps will be copied.
        Returns: MPS
            The copy of self.
        '''
        ms=[m.copy(copy_data=copy_data) for m in self]
        Lambda=None if self.Lambda is None else self.Lambda.copy(copy_data=copy_data)
        return MPS(ms=ms,Lambda=Lambda,cut=self.cut)

    def relabel(self,news,olds=None):
        '''
        Change the labels of the MPS.
        Parameters:
            news: list of 3-tuples of Label
                The new labels of the MPS.
            olds: list of 3-tuples of Label, optional
                The old labels of the MPS.
        '''
        if olds is None:
            assert len(news)==self.nsite
            for m,new in zip(self,news):
                m.relabel(new)
        else:
            table=self.table
            assert len(news)==len(olds)
            for new,old in zip(news,olds):
                self[table[old[MPS.S]]].relabel(new)

    @staticmethod
    def overlap(mps1,mps2):
        '''
        The overlap between two mps.
        Parameters:
            mps1,mps2: MPS
                The MPS between which the overlap is calculated.
        Returns: number
            The overlap.
        '''
        def reset_and_protect(mps):
            if mps.cut is None:
                m,Lambda=None,None
            elif mps.cut<mps.nsite:
                m,Lambda=mps[mps.cut],mps.Lambda
                mps._reset_(merge='R',reset=None)
            else:
                m,Lambda=mps[mps.cut-1],mps.Lambda
                mps._reset_(merge='L',reset=None)
            return m,Lambda
        u1,Lambda1=reset_and_protect(mps1)
        u2,Lambda2=reset_and_protect(mps2)
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
        mps1._set_ABL_(u1,Lambda1)
        mps2._set_ABL_(u2,Lambda2)
        return np.asarray(contract(*result,sequence='sequential'))

    # NOT DEBUGED COED
    def level_down(self,degfres,n=1):
        '''
        Construt a new mps with the physical indices which are n-level down in the degfres.
        Parameters:
            degfres: DegFreTree
                The tree of the physical degrees of freedom.
            n: integer, optional
                The degree of level to go down.
        Returns: MPS
            The new MPS.
        '''
        assert n>=0
        assert self.mode==degfres.mode
        result=self if n==0 else copy(self)
        table=result.table
        for count in xrange(n):
            level=degfres.level(next(iter(table)).identifier)
            if level==1:
                raise ValueError("MPS level_down error: at the %s-th loop, level(%s) is already at the bottom."%(count,level))
            result._reset_(reset=None)
            olds=degfres.labels(layer=degfres.layers[level-1],full_labels=False)
            news=degfres.labels(layer=degfres.layers[level-2],full_labels=True)
            ms=[None]*len(news)
            if result.mode=='NB':
                for k,parent in enumerate(news.keys()):
                    for i,child in enumerate(degfres.children(parent)):
                        if i==0:
                            ms[k]=np.asarray(result[table[olds[child]]])
                        else:
                            ms[k]=np.einsum('ijk,klm->ijlm',ms[k],np.asarray(result[table[olds[child]]]))
                            shape=ms[k].shape
                            ms[k]=ms[k].reshape((shape[0],-1,shape[-1]))
                result=MPS(mode='NB',ms=ms,labels=news.values(),Lambda=None,cut=None)
            else:
                labels=[]
                qncs=degfres.qnc_evolutions(layer=degfres.layers[level-2])
                for k,L,S,R in enumerate(news.values()):
                    for i,child,sqnc in enumerate(zip(degfres.children(S.identifier),qncs[S.identifier])):
                        pos=table[olds[child]]
                        if i==0:
                            ms[k]=np.asarray(result[pos])
                            lqnc=result[pos].labels[MPS.L].qnc
                        else:
                            ms[k]=np.einsum('ijk,klm->ijlm',ms[k],np.asarray(result[pos]))
                            shape=ms[k].shape
                            ms[k]=sqnc.reorder(ms[k].reshape((shape[0],-1,shape[-1])),axes=[1])
                        if i==len(degfres.children(S.identifier)):
                            rqnc=result[pos].labels[MPS.R].qnc
                    labels.append((L.replace(qnc=lqnc),S.replace(qnc=sqnc),R.replace(qnc=rqnc)))
                result=MPS(mode='QN',ms=ms,labels=labels,Lambda=None,cut=None)
        return result

    def level_up(self,degfres,n=1,nmax=None,tol=None):
        '''
        Construt a new mps with the physical indices which are n-level up in the degfres.
        Parameters:
            degfres: DegFreTree
                The tree of the physical degrees of freedom.
            n: integer, optional
                The degree of level to go up.
            nmax: integer, optional
                The maximum number of singular values to be kept.
            tol: float64, optional
                The tolerance of the singular values.
        Returns: MPS
            The new MPS.
        '''
        assert n>=0
        assert self.mode==degfres.mode
        result=self if n==0 else copy(self)
        table=result.table
        for count in xrange(n):
            level=degfres.level(next(iter(table)).identifier)
            if level==len(degfres.layers):
                raise ValueError("MPS level_up error: at the %s-th loop, level(%s) is already at the top."%(count,level))
            labels=degfres.labels(layer=degfres.layers[level],full_labels=True)
            assert result.Lambda is not None
            result>>=(result.nsite-result.cut,nmax,tol)
            ms,ls=[],[]
            for i,m in enumerate(reversed(result)):
                if i==0:
                    L,S,R=m.labels[MPS.L],m.labels[MPS.S],m.labels[MPS.R]
                    m=np.einsum('ijk,k->ijk',np.asarray(m),np.asarray(result.Lambda))
                else:
                    L,S,R=m.labels[MPS.L],m.labels[MPS.S],ls[0][MPS.L]
                    m=np.einsum('ijk,kl,l->ijl',np.asarray(m),u,s)
                children=degfres.children(S.identifier)
                expansion=[degfres[child] for child in children]
                u,s,vs,qncs=expanded_svd(m,expansion=expansion,L=L.qnc,S=S.qnc,R=R.qnc,mode='vs',nmax=nmax,tol=tol)
                for k in reversed(xrange(len(children))):
                    Left,Site,Right=labels[children[k]]
                    ms.insert(0,vs[k])
                    ls.insert(0,(Left.replace(qnc=qncs[k]),Site,Right.replace(qnc=qncs[k+1] if k+1<len(children) else R.qnc)))
            result=MPS(mode=result.mode,ms=ms,labels=ls,Lambda=u.dot(s),cut=0)
        return result

class Vidal(object):
    '''
    The Vidal canonical matrix product state.
    Attributes:
        Gammas: list of Tensor
            The Gamma matrices on the site.
        Lambdas: list of Tensor
            The Lambda matrices (singular values) on the link.
    '''
    L,S,R=0,1,2

    def __init__(self,Gammas,Lambdas,labels=None):
        '''
        Constructor.
        Parameters:
            Gammas: list of 3d ndarray/Tensor
                The Gamma matrices on the site.
            Lamdas: list of 1d ndarray/Tensor
                The Lambda matrices (singular values) on the link.
            labels: list of 3 tuples, optional
                The labels of the axis of the Gamma matrices.
                Its length should be equal to that of Gammas.
                For each label in labels, 
                    label[0],label[1],label[2]: any hashable object
                        The left link / site / right link label of the matrix.
        '''
        assert len(Gammas)==len(Lambdas)+1
        self.Gammas=[]
        self.Lambdas=[]
        if labels is None:
            assert len(Gammas)==len(labels)
            buff=[]
            for i,(Gamma,label) in enumerate(zip(Gammas,labels)):
                assert Gamma.ndim==3
                if i<len(Gammas)-1:
                    buff.append(R)
                self.Gammas.append(Tensor(Gamma,labels=list(label)))
            for Lambda,label in zip(Lambdas,buff):
                assert Lambda.ndim==1
                self.Lambdas.append(Tensor(Lambda,labels=[label]))
        else:
            for Gamma in Gammas:
                assert isinstance(Gamma,Tensor)
                assert Gamma.ndim==3
                self.Gammas.append(Gamma)
            for Lambda in Lambdas:
                assert isinstance(Lambda,Tensor)
                assert Lambda.ndim==1
                self.Lambdas.append(Lambda)

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
        Returns: 1d ndarray
            The corresponding normal representation of the state.
        '''
        result=None
        for i,Gamma in enumerate(self.Gammas):
            if result is None:
                result=Gamma
            else:
                result=contract(result,self.Lambdas[i-1],Gamma)
        return np.asarray(result).ravel()

    def to_mixed(self,cut):
        '''
        Convert to the mixed MPS representation.
        Parameters:
            cut: integer
                The index of the connecting link.
        Retruns: MPS
            The corresponding mixed MPS.
        '''
        ms,labels,Lambda=[],[],None
        shape=[1]*3
        shape[self.S]=-1
        for i,Gamma in enumerate(self.Gammas):
            L,S,R=Gamma.labels[self.L],Gamma.labels[self.S],Gamma.labels[self.R]
            labels.append((L,S,R))
            if i<cut:
                if i==0:
                    ms.append(np.asarray(Gamma))
                else:
                    ms.append(np.asarray(Gamma)*np.asarray(self.Lambdas[i-1]).reshape(shape))
            else:
                if i>0 and i==cut:
                    Lambda=np.asarray(self.Lambdas[i-1])
                if i<len(self.Lambdas):
                    ms.append(np.asarray(Gamma)*np.asarray(self.Lambdas[i]).reshape(shape))
                else:
                    ms.append(np.asarray(Gamma))
        return MPS(ms=ms,labels=labels,Lambda=Lambda,cut=cut)
