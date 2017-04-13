'''
Matrix product operator, including:
1) classes: OptStr, MPO
'''

__all__=['OptStr','MPO']

import numpy as np
import HamiltonianPy.Misc as hm
from numpy.linalg import norm
from collections import OrderedDict
from HamiltonianPy import QuantumNumbers,OperatorF,OperatorS,CREATION
from Tensor import Tensor,Label,contract
from MPS import MPS
from copy import copy

class OptStr(list):
    '''
    Operator string, a special kind of matrix product operator, with the virtual legs of the matrices always one dimensional and thus omitted.
        For each of its elements: 2d Tensor
            The matrices of the mpo.
    '''

    def __init__(self,ms,sites=None):
        '''
        Constructor.
        Parameters:
            ms: 2d ndarray/Tensor
                The matrices of the mpo.
            sites: list of Label, optional
                The site labels of the mpo.
        '''
        if sites is None:
            for m in ms:
                assert m.ndim==2 and m.labels[0]==m.labels[1].prime
                self.append(m)
        else:
            assert len(ms)==len(sites)
            for m,site in zip(ms,sites):
                assert m.ndim==2
                self.append(Tensor(m,labels=[site.prime,site]))

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return '\n'.join(str(m) for m in self)

    @staticmethod
    def from_operator(operator,degfres,layer):
        '''
        Constructor, which converts an operator to an optstr.
        Parameters:
            operator: OperatorS, OperatorF
                The operator to be converted to an optstr.
            degfres: DegFreTree
                The degfretree of the system.
            layer: integer or tuple of string
                The layer where the converted optstr lives.
        Returns: OptStr
            The corresponding OptStr.
        '''
        assert type(operator) in (OperatorS,OperatorF)
        layer=degfres.layers[layer] if type(layer) in (int,long) else layer
        if type(operator) is OperatorS:
            ms=[]
            table,sites=degfres.table(degfres.layers[-1]),degfres.labels(degfres.layers[-1],'S')
            for i,(index,matrix) in enumerate(zip(operator.indices,operator.spins)):
                pos=table[index]
                ms.append(Tensor(matrix*operator.value if i==0 else matrix,labels=[sites[pos].prime,sites[pos]]))
            return OptStr(sorted(ms,key=lambda m:table[m.labels[1].identifier])).relayer(degfres,layer)
        else:
            length=len(operator.indices)
            assert length%2==0
            table=degfres.table(degfres.layers[-1])
            permutation=sorted(range(length),key=lambda k:table[operator.indices[k].replace(nambu=None)])
            groups,counts=OrderedDict(),[]
            for k in permutation:
                leaf=table[operator.indices[k].replace(nambu=None)]
                m=np.array([[0.0,0.0],[1.0,0.0]]) if operator.indices[k].nambu==CREATION else np.array([[0.0,1.0],[0.0,0.0]])
                if leaf in groups:
                    counts[-1]+=1
                    groups[leaf]=groups[leaf].dot(m)
                else:
                    counts.append(1)
                    groups[leaf]=m
            ms=[]
            keys=groups.keys()
            sites=degfres.labels(degfres.layers[-1],'S')
            zmatrix=np.array([[1.0,0.0],[0.0,-1.0]])
            for leaf in xrange(keys[0],keys[-1]+1):
                labels=[sites[leaf].prime,sites[leaf]]
                if leaf in groups:
                    assert counts[0] in (1,2)
                    length-=counts.pop(0)
                    ms.append(Tensor(groups[leaf] if length%2==0 else groups[leaf].dot(zmatrix),labels=labels))
                elif length%2!=0:
                    ms.append(Tensor(zmatrix,labels=labels))
            ms[0]=ms[0]*operator.value*hm.parity(permutation)
            return OptStr(ms=sorted(ms,key=lambda m: m.labels[1].identifier)).relayer(degfres,layer)

    def __imul__(self,other):
        '''
        Overloaded self-multiplication(*=) operator, which supports the self-multiplication by a scalar.
        '''
        self[0]*=other
        return self

    def __mul__(self,other):
        '''
        Overloaded multiplication(*) operator, which supports the multiplication of an optstr with a scalar.
        '''
        result=copy(self)
        result[0]=result[0]*other
        return result

    def __rmul__(self,other):
        '''
        Overloaded multiplication(*) operator, which supports the multiplication of a scalar with an optstr.
        '''
        return self*other

    def __idiv__(self,other):
        '''
        Overloaded self-division(/=) operator, which supports the self-division by a scalar.
        '''
        return self.__imul__(1.0/other)

    def __div__(self,other):
        '''
        Overloaded division(/) operator, which supports the division of an optstr by a scalar.
        '''
        return self.__mul__(1.0/other)

    def overlap(self,mps1,mps2):
        '''
        The overlap of an optstr between two mpses.
        Parameters:
            mps1,mps2: MPS
                The two matrix product states between which the overlap of an optstr is calculated.
                Both mpses should be kets because the complex conjugate of the first mps is always taken to calculate the overlap in this function.
        Returns: number
            The overlap.
        '''
        reset_and_protect=lambda mps,start: mps._merge_ABL_('R') if mps.cut==start else mps._merge_ABL_('L')
        poses={m.labels[1]:mps1.table[m.labels[1]] for m in self}
        ms=sorted(self,key=lambda m:poses[m.labels[1]])
        if mps1 is mps2:
            start,stop,count=poses[ms[0].labels[1]],poses[ms[-1].labels[1]]+1,0
            if mps1.cut<start or mps1.cut>stop:
                if mps1.cut<start:
                    mps1>>=start-mps1.cut
                else:
                    mps1<<=mps1.cut-stop
            m,Lambda=reset_and_protect(mps1,start)
        else:
            start,stop,count=0,mps1.nsite,0
            m1,Lambda1=reset_and_protect(mps1,start)
            m2,Lambda2=reset_and_protect(mps2,start)
        result=Tensor(1.0,labels=[])
        for i,(u1,u2) in enumerate(zip(mps1[start:stop],mps2[start:stop])):
            u1=u1.copy(copy_data=False).conjugate()
            L1,S1,R1=u1.labels
            L2,S2,R2=u2.labels
            assert L1==L2 and S1==S2 and R1==R2
            news,olds=[L1.prime,S1.prime,R1.prime],[L1,S1,R1]
            if i==0:
                news.remove(L1.prime)
                olds.remove(L1)
            if i==stop-start-1:
                news.remove(R1.prime)
                olds.remove(R1)
            if S1 in poses:
                u1.relabel(news=news,olds=olds)
                result=contract([result,u1,ms[count],u2],engine='einsum',sequence='sequential')
                count+=1
            else:
                news.remove(S1.prime)
                olds.remove(S1)
                u1.relabel(news=news,olds=olds)
                result=contract([result,u1,u2],engine='einsum',sequence='sequential')
        if mps1 is mps2:
            mps1._set_ABL_(m,Lambda)
        else:
            mps1._set_ABL_(m1,Lambda1)
            mps2._set_ABL_(m2,Lambda2)
        return np.asarray(result)

    def to_mpo(self,degfres):
        '''
        Convert an optstr to the full-formated mpo.
        Parameters:
            degfres: DegFreTree
                The tree of the site degrees of freedom.
        Returns: MPO
            The corresponding MPO.
        '''
        type=degfres[next(iter(next(iter(self)).labels)).identifier].type if degfres.mode=='QN' else None
        layer=degfres.layers[degfres.level(next(iter(self)).labels[1].identifier)-1]
        table,sites,bonds=degfres.table(layer),degfres.labels(layer,'S'),degfres.labels(layer,'O')
        poses=set(table[m.labels[1].identifier] for m in self)
        ms,count=[],0
        for pos in xrange(len(sites)):
            L,U,D,R=copy(bonds[pos]),sites[pos].prime,sites[pos],copy(bonds[pos+1])
            ndegfre=degfres.ndegfre(U.identifier)
            if degfres.mode=='QN':
                U,D=U.replace(qns=None),D.replace(qns=None)
                lqns,sqns=QuantumNumbers.mono(type.zeros()) if pos==0 else ms[-1].labels[MPO.R].qns,sites[pos].qns
            if pos in poses:
                ms.append(Tensor(np.asarray(self[count]).reshape((1,ndegfre,ndegfre,1)),labels=[L,U,D,R]))
                count+=1
            else:
                ms.append(Tensor(np.identity(len(sqns)).reshape((1,ndegfre,ndegfre,1)),labels=[L,U,D,R]))
            if degfres.mode=='QN':
                ms[-1].qng(axes=[MPO.L,MPO.U,MPO.D],qnses=[lqns,sqns,sqns],signs='++-')
        return MPO(ms)

    def relayer(self,degfres,layer):
        '''
        Construt a new optstr with the site labels living on a specific layer of degfres.
        Parameters:
            degfres: DegFreTree
                The tree of the site degrees of freedom.
            layer: integer/tuple-of-string
                The layer where the site labels live.
        Returns: OptStr
            The new optstr.
        '''
        new=layer if type(layer) in (int,long) else degfres.layers.index(layer)
        old=degfres.level(next(iter(next(iter(self)).labels)).identifier)-1
        assert new>=0 and new<len(degfres.layers) and old>=new
        if old==new:
            return copy(self)
        else:
            poses={}
            for pos,m in enumerate(self):
                index=m.labels[1].identifier
                ancestor=degfres.ancestor(index,generation=old-new)
                if ancestor in poses:
                    poses[ancestor][index]=pos
                else:
                    poses[ancestor]={index:pos}
            ms=[]
            olayer,nlayer=degfres.layers[old],degfres.layers[new]
            otable,ntable=degfres.table(olayer),degfres.table(nlayer)
            sites=degfres.labels(nlayer,'S')
            for ancestor in sorted(poses.keys(),key=ntable.get):
                m=1.0
                for index in degfres.descendants(ancestor,old-new):
                    if index in poses[ancestor]:
                        m=np.kron(m,np.asarray(self[poses[ancestor][index]]))
                    else:
                        m=np.kron(m,np.identity(degfres.ndegfre(index)))
                pos=ntable[ancestor]
                ms.append(Tensor(m,labels=[sites[pos].prime,sites[pos]]))
            return OptStr(ms)

class MPO(list):
    '''
    Matrix product operator.
        For each of its elements: 2d Tensor
            The matrices of the mpo.
    '''
    L,U,D,R=0,1,2,3

    def __init__(self,ms,sites=None,bonds=None):
        '''
        Constructor.
        Parameters:
            ms: list of 4d ndarray/Tensor
                The matrices of the mpo.
            sites: list of Label, optional
                The site labels of the mpo.
            bonds: list of Label, optional
                The bond labels of the mpo.
        '''
        assert (sites is None)==(bonds is None)
        if sites is None:
            for m in ms:
                assert m.ndim==4 and m.labels[MPO.U]==m.labels[MPO.D].prime
                self.append(m)
        else:
            assert len(ms)==len(sites) and len(ms)==len(bonds)-1
            for i,m in enumerate(ms):
                assert m.ndim==4
                self.append(Tensor(m,labels=[bonds[i],sites[i].prime,sites[i],bonds[i+1]]))

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return '\n'.join(str(m) for m in self)

    @property
    def nsite(self):
        '''
        The number of total sites.
        '''
        return len(self)

    @property
    def sites(self):
        '''
        The site labels of the mpo.
        '''
        return [m.labels[MPO.D] for m in self]

    @property
    def bonds(self):
        '''
        The bond labels of the mpo.
        '''
        result=[]
        for i,m in enumerate(self):
            if i==0: result.append(m.labels[MPO.L])
            result.append(m.labels[MPO.R])
        return result

    @property
    def matrix(self):
        '''
        The normal matrix representation of the mpo.
        '''
        ls,rs,L,R=[],[],1,1
        for i,m in enumerate(self):
            ls.append(m.labels[MPO.U])
            rs.append(m.labels[MPO.D])
            L*=ls[-1].dim
            R*=rs[-1].dim
            if i==0:
                result=m
            else:
                result=contract([result,m],engine='tensordot')
        return np.asarray(result.transpose(axes=[self[0].labels[MPO.L]]+ls+rs+[self[-1].labels[MPO.R]])).reshape((L,R))

    def _mul_mpo_(self,other):
        '''
        The multiplication of two mpos.
        Parameters:
            other: MPO
                The other mpo.
        Returns: MPO
            The product.
        '''
        assert self.nsite==other.nsite
        ms=[]
        for i,(m1,m2) in enumerate(zip(self,other)):
            assert m1.labels==m2.labels
            m1,m2=m1.copy(copy_data=False),m2.copy(copy_data=False)
            L1,U1,D1,R1=m1.labels
            L2,U2,D2,R2=m2.labels
            L=L1.replace(qns=QuantumNumbers.kron([L1.qns,L2.qns]) if L1.qnon else L1.qns*L2.qns)
            R=R1.replace(qns=QuantumNumbers.kron([R1.qns,R2.qns]) if R1.qnon else R1.qns*R2.qns)
            s=Label('__MPO_MUL__',qns=U1.qns)
            l1,r1=Label('__MPO_MUL_L1__',qns=L1.qns),Label('__MPO_MUL_R1__',qns=R1.qns)
            l2,r2=Label('__MPO_MUL_L2__',qns=L2.qns),Label('__MPO_MUL_R2__',qns=R2.qns)
            m1.relabel(olds=[L1,D1,R1],news=[l1,s,r1])
            m2.relabel(olds=[L2,U2,R2],news=[l2,s,r2])
            ms.append(contract([m1,m2],engine='tensordot').transpose((l1,l2,U1,D2,r1,r2)).merge((([l1,l2],L)),([r1,r2],R)))
        return MPO(ms)

    def _mul_mps_(self,other):
        '''
        The multiplication of an mpo and an mps.
        Parameters:
            other: MPS
                The mps.
        Returns: MPS
            The product.
        '''
        assert self.nsite==other.nsite
        u,Lambda=other._merge_ABL_()
        ms=[]
        for i,(m1,m2) in enumerate(zip(self,other)):
            L1,U1,D1,R1=m1.labels
            L2,S,R2=m2.labels
            assert S==D1
            L=L2.replace(qns=QuantumNumbers.kron([L1.qns,L2.qns]) if L1.qnon else L1.qns*L2.qns)
            R=R2.replace(qns=QuantumNumbers.kron([R1.qns,R2.qns]) if R1.qnon else R1.qns*R2.qns)
            m=contract([m1,m2],engine='tensordot').transpose((L1,L2,U1,R1,R2)).merge(([L1,L2],L),([R1,R2],R))
            m.relabel(olds=[U1],news=[S])
            ms.append(m)
        other._set_ABL_(u,Lambda)
        return MPS(mode=other.mode,ms=ms)

    def __iadd__(self,other):
        '''
        Overloaded self-addition(+=) operator.
        '''
        return self+other

    def __add__(self,other):
        '''
        Overloaded addition(+) operator, which supports the addition of two mpos.
        '''
        assert self.nsite==other.nsite
        ms=[]
        for i,(m1,m2) in enumerate(zip(self,other)):
            assert m1.labels==m2.labels
            labels=[label.replace(qns=None) for label in m1.labels]
            axes=[MPO.L,MPO.U,MPO.D] if i==0 else ([MPO.U,MPO.D,MPO.R] if i==self.nsite-1 else [MPO.U,MPO.D])
            ms.append(Tensor.directsum([m1,m2],labels=labels,axes=axes))
        return MPO(ms)

    def __pos__(self):
        '''
        Overloaded positive(+) operator.
        '''
        return copy(self)

    def __sub__(self,other):
        '''
        Overloaded subtraction(-) operator, which supports the subtraction of two mpos.
        '''
        return self+other*(-1)

    def __neg__(self):
        '''
        Overloaded negative(-) operator.
        '''
        return self*(-1)

    def __imul__(self,other):
        '''
        Overloaded self-multiplication(*=) operator, which supports the self-multiplication by a scalar.
        '''
        if isinstance(other,MPO):
            return self._mul_mpo_(other)
        else:
            self[0]*=other
            return self

    def __mul__(self,other):
        '''
        Overloaded multiplication(*) operator, which supports the multiplication of an mpo with a scalar.
        '''
        if isinstance(other,MPO):
            result=self._mul_mpo_(other)
        elif isinstance(other,MPS):
            result=self._mul_mps_(other)
        else:
            result=copy(self)
            result[0]=result[0]*other
        return result

    def __rmul__(self,other):
        '''
        Overloaded multiplication(*) operator, which supports the multiplication of a scalar with an mpo.
        '''
        return self*other

    def __idiv__(self,other):
        '''
        Overloaded self-division(/=) operator, which supports the self-division by a scalar.
        '''
        return self.__imul__(1.0/other)

    def __div__(self,other):
        '''
        Overloaded division(/) operator, which supports the division of an mpo by a scalar.
        '''
        return self.__mul__(1.0/other)

    def overlap(self,mps1,mps2):
        '''
        The overlap of an mpo between two mpses.
        Parameters:
            mps1, mps2: MPS
                The two matrix product states between which the overlap of an mpo is calculated.
                Both mpses should be kets because the complex conjugate of the first mps is always taken to calculate the overlap in this function.
        Returns: number
            The overlap.
        '''
        assert self.nsite==mps1.nsite and self.nsite==mps2.nsite
        if mps1 is mps2:
            u,Lambda=mps1._merge_ABL_()
        else:
            u1,Lambda1=mps1._merge_ABL_()
            u2,Lambda2=mps2._merge_ABL_()
        result=Tensor(1.0,labels=[])
        for i,(mpo,m1,m2) in enumerate(zip(self,mps1,mps2)):
            m1=m1.copy(copy_data=False).conjugate()
            L1,S1,R1=m1.labels
            L2,S2,R2=m2.labels
            assert L1==L2 and S1==S2 and R1==R2
            olds,news=[L1,S1,R1],[L1.prime,S1.prime,R1.prime]
            if i==0:
                olds.remove(L1)
                news.remove(L1.prime)
            if i==self.nsite-1:
                olds.remove(R1)
                news.remove(R1.prime)
            m1.relabel(olds=olds,news=news)
            result=contract([result,m1,m2,mpo],engine='tensordot')
        if mps1 is mps2:
            mps1._set_ABL_(u,Lambda)
        else:
            mps1._set_ABL_(u1,Lambda1)
            mps2._set_ABL_(u2,Lambda2)
        assert result.shape==(1,1)
        return result[0,0]

    def compress(self,nsweep=1,method='dpl',options={}):
        '''
        Compress the mpo.
        Parameters:
            nsweep: integer, optional
                The number of sweeps to compress the mpo.
            method: 'svd', 'dpl' or 'dln'
                The method used to compress the mpo.
            options: dict, optional
                The options used to compress the mpo.
        Returns: MPO
            The compressed mpo.
        '''
        assert method in ('svd','dpl','dln')
        if method=='svd':
            tol=options.get('tol',hm.TOL)
            for sweep in xrange(nsweep):
                for i,m in enumerate(self):
                    if i<self.nsite-1:
                        L,U,D,R=m.labels
                        u,s,v=m.svd(row=[L,U,D],new=R.prime,col=[R],row_signs='++-',col_signs='+',tol=tol)
                        self[i]=contract([u,s],engine='einsum',reserve=s.labels)
                        self[i+1]=contract([v,self[i+1]],engine='tensordot')
                        self[i].relabel(olds=s.labels,news=[R.replace(qns=s.labels[0].qns)])
                        self[i+1].relabel(olds=s.labels,news=[R.replace(qns=s.labels[0].qns)])
                for i,m in enumerate(reversed(self)):
                    if i<self.nsite-1:
                        L,U,D,R=m.labels
                        u,s,v=m.svd(row=[L],new=L.prime,col=[U,D,R],row_signs='+',col_signs='-++',tol=tol)
                        self[-1-i]=contract([s,v],engine='einsum',reserve=s.labels)
                        self[-2-i]=contract([self[-2-i],u],engine='tensordot')
                        self[-1-i].relabel(olds=s.labels,news=[L.replace(qns=s.labels[0].qns)])
                        self[-2-i].relabel(olds=s.labels,news=[L.replace(qns=s.labels[0].qns)])
                for m in self:
                    m[np.abs(m)<tol]=0.0
        elif method=='dpl':
            zero=options.get('zero',10**-8)
            tol=options.get('tol',10**-6)
            for sweep in xrange(nsweep):
                for i,m in enumerate(reversed(self)):
                    if i<self.nsite-1:
                        L,U,D,R=m.labels
                        T,M=m.deparallelization(row=[L],new=L.prime,col=[U,D,R],mode='R',zero=zero,tol=tol)
                        self[-1-i]=M
                        self[-2-i]=contract([self[-2-i],T],engine='tensordot')
                        self[-1-i].relabel(olds=[L.prime],news=[L.replace(qns=T.labels[1].qns)])
                        self[-2-i].relabel(olds=[L.prime],news=[L.replace(qns=T.labels[1].qns)])
                for i,m in enumerate(self):
                    if i<self.nsite-1:
                        L,U,D,R=m.labels
                        M,T=m.deparallelization(row=[L,U,D],new=R.prime,col=[R],mode='C',zero=zero,tol=tol)
                        self[i]=M
                        self[i+1]=contract([T,self[i+1]],engine='tensordot')
                        self[i].relabel(olds=[R.prime],news=[R.replace(qns=T.labels[0].qns)])
                        self[i+1].relabel(olds=[R.prime],news=[R.replace(qns=T.labels[0].qns)])
        else:
            pass

    def relayer(self,degfres,layer,nmax=None,tol=None):
        '''
        Construt a new mpo with the site labels living on a specific layer of degfres.
        Parameters:
            degfres: DegFreTree
                The tree of the site degrees of freedom.
            layer: integer/tuple-of-string
                The layer where the site labels live.
            nmax: integer, optional
                The maximum number of singular values to be kept.
            tol: np.float64, optional
                The tolerance of the singular values.
        Returns: MPO
            The new mpo.
        '''
        new=layer if type(layer) in (int,long) else degfres.layers.index(layer)
        old=degfres.level(next(iter(self)).labels[MPO.U].identifier)-1
        assert new>=0 and new<len(degfres.layers)
        if new==old:
            return copy(self)
        else:
            olayer,nlayer=degfres.layers[old],degfres.layers[new]
            sites,bonds=degfres.labels(nlayer,'S'),degfres.labels(nlayer,'O')
            Ms=[]
            if new<old:
                table=degfres.table(olayer)
                for i,site in enumerate(sites):
                    ms,ups,dws=[],[],[]
                    for index in degfres.descendants(site.identifier,generation=old-new):
                        m=self[table[index]]
                        ms.append(m)
                        ups.append(m.labels[MPO.U])
                        dws.append(m.labels[MPO.D])
                    M=contract(ms,engine='tensordot')
                    o1,o2=M.labels[0],M.labels[-1]
                    n1,n2=bonds[i].replace(qns=o1.qns),bonds[i+1].replace(qns=o2.qns)
                    M.relabel(olds=[o1,o2],news=[n1,n2])
                    Ms.append(M.transpose([n1]+ups+dws+[n2]).merge((ups,site.prime),(dws,site)))
            else:
                table=degfres.table(nlayer)
                for i,m in enumerate(self):
                    if i>0: m=contract([s,v,m],engine='einsum',reserve=s.labels)
                    L,U,D,R=m.labels
                    indices=degfres.descendants(D.identifier,generation=new-old)
                    start,stop=table[indices[0]],table[indices[-1]]+1
                    ups,dws,orders,labels,qnses=[],[],[],[],[]
                    for i,site in enumerate(sites[start:stop]):
                        ups.append(site.prime)
                        dws.append(site)
                        orders.append(ups[-1])
                        orders.append(dws[-1])
                        qns=QuantumNumbers.kron([site.qns]*2,signs='+-') if site.qnon else site.dim**2
                        labels.append(Label('__MPO_RELAYER_%s__'%i,qns=qns))
                        qnses.append(qns)
                    S=Label('__MPO_RELAYER__',qns=QuantumNumbers.kron(qnses) if U.qnon else np.product(qnses))
                    m=m.split((U,ups),(D,dws)).transpose([L]+orders+[R]).merge((orders,S))
                    us,s,v=m.expanded_svd(L=[L],S=S,R=[R],E=labels,I=bonds[start+1:stop+1],cut=stop-start,nmax=nmax,tol=tol)
                    for u,up,dw,label in zip(us,ups,dws,labels):
                        Ms.append(u.split((label,[up,dw])))
                Ms[-1]=contract([Ms[-1],s,v],engine='einsum')
                Ms[+0].relabel(olds=[MPO.L],news=[bonds[+0].replace(qns=Ms[+0].labels[MPO.L].qns)])
                Ms[-1].relabel(olds=[MPO.R],news=[bonds[-1].replace(qns=Ms[-1].labels[MPO.R].qns)])
            return MPO(Ms)
