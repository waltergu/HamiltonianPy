'''
=======================
Matrix product operator
=======================

Matrix product operator, including:
    * classes: Opt, OptStr, OptMPO, MPO
'''

__all__=['Opt','OptStr','OptMPO','MPO']

import numpy as np
import itertools as it
from numpy.linalg import norm
from collections import OrderedDict
from HamiltonianPy import QuantumNumbers,Operator,FOperator,SOperator,JWBosonization
from HamiltonianPy.Misc import TOL,Arithmetic
from Tensor import Tensor,Label,contract
from MPS import MPS
from copy import copy

class Opt(Operator.Operator):
    '''
    A single site operator.

    Attributes
    ----------
    site : Label
        The site label of the single site operator.
    tag : any hashable object
        The tag of the single site operator.
    matrix : 2d ndarray
        The matrix of the single site operator.
    '''

    def __init__(self,value,site,tag,matrix):
        '''
        Constructor.

        Parameters
        ----------
        value : number
            The overall coefficient of the single site operator.
        site : Label
            The site label of the single site operator.
        tag : any hashable object
            The tag of the single operator.
        matrix : 2d ndarray
            The matrix of the single site operator.
        '''
        assert matrix.ndim==2
        dtype=np.find_common_type([np.asarray(value).dtype,np.asarray(matrix).dtype],[])
        self.value=np.array(value,dtype=dtype)
        self.site=site
        self.tag=tag
        self.matrix=np.asarray(matrix,dtype=dtype)

    @staticmethod
    def identity(site,dtype=np.float64):
        '''
        Construt an identity single site operator.

        Parameters
        ----------
        site : Label
            The site label.
        dtype : np.float64, np.complex128, etc, optional
            The data type of the identity.

        Returns
        -------
        Opt
            The identity single site operator.
        '''
        return Opt(1.0,site,'i',np.identity(site.dim,dtype=dtype))

    @staticmethod
    def zero(site,dtype=np.float64):
        '''
        Construt a zero single site operator.

        Parameters
        ----------
        site : Label
            The site label.
        dtype : np.float64, np.complex128, etc, optional
            The data type of the zero.

        Returns
        -------
        Opt
            The zero single site operator.
        '''
        return Opt(1.0,site,'0',np.zeros((site.dim,site.dim),dtype=dtype))

    @property
    def tensor(self):
        '''
        The tensor representation of the single site operator.
        '''
        return Tensor(self.value*self.matrix,labels=[self.site.prime,self.site])

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return 'Opt(value=%s,site=%s,tag=%s,matrix=\n%s)'%(self.value,self.site,self.tag,self.matrix)

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return "%s(%s)"%(self.value,self.tag) if norm(self.value-1.0)>0 else str(self.tag)

    def __iadd__(self,other):
        '''
        Overloaded self addition(+) operator, which supports the self addition by an instance of Opt.
        '''
        assert other.site==self.site
        if self.tag=='0':
            return other
        elif other.tag=='0':
            return self
        else:
            if self.tag==other.tag:
                self.value+=other.value
            else:
                self.value=np.array(1.0,dtype=np.find_common_type([self.value.dtype,other.value.dtype],[]))
                self.tag='%s(%s)+%s(%s)'%(self.value,self.tag,other.value,other.tag)
                self.matrix*=self.value
                self.matrix+=other.value*other.matrix
            return self

    def __add__(self,other):
        '''
        Overloaded left addition(+) operator, which supports the left addition by an instance of Opt.
        '''
        assert other.site==self.site
        if other.tag=='0':
            return self
        elif self.tag=='0':
            return other
        else:
            if self.tag==other.tag:
                return Opt(self.value+other.value,self.site,self.tag,self.matrix)
            else:
                return Opt(1.0,self.site,'%s(%s)+%s(%s)'%(self.value,self.tag,other.value,other.tag,self.value*self.matrix+other.value*other.matrix))

class OptStr(Arithmetic,list):
    '''
    Operator string, a list of single site operators.
    '''

    def __init__(self,opts):
        '''
        Constructor.

        Parameters
        ----------
        opts : list of Opt
            The single site operators of the optstr.
        '''
        self.extend(opts)

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return '\n'.join(str(opt) for opt in self)

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return '\n'.join('%s%s%s'%(opt.value,repr(opt.site),opt.tag) for opt in self)

    @staticmethod
    def from_operator(operator,degfres,layer=0):
        '''
        Constructor, which converts an operator to an optstr.

        Parameters
        ----------
        operator : SOperator, FOperator
            The operator to be converted to an optstr.
        degfres : DegFreTree
            The degfretree of the system.
        layer : integer/tuple-of-string, optional
            The layer where the converted optstr lives.

        Returns
        -------
        OptStr
            The corresponding OptStr.
        '''
        assert type(operator) in (SOperator,FOperator)
        dtype,layer=np.array(operator.value).dtype,degfres.layers[layer] if type(layer) in (int,long) else layer
        table,sites=degfres.table(degfres.layers[-1]),degfres.labels('S',degfres.layers[-1])
        operator=operator if type(operator) is SOperator else JWBosonization(operator,table)
        opts=[]
        permutation=sorted(range(len(operator.indices)),key=lambda k: table[operator.indices[k]])
        for i,k in enumerate(permutation):
            index,matrix=operator.indices[k],operator.spins[k]
            opts.append(Opt(operator.value if i==0 else 1.0,sites[table[index]],matrix.tag,matrix))
        return OptStr(opts).relayer(degfres,layer)

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

    def overlap(self,mps1,mps2):
        '''
        The overlap of an optstr between two mpses.

        Parameters
        ----------
        mps1,mps2 : MPS
            The two matrix product states between which the overlap of an optstr is calculated.

        Returns
        -------
        number
            The overlap.

        Notes
        -----
        Both mpses should be kets since in this function the complex conjugate of the first mps is always taken during the calculation.
        '''
        reset_and_protect=lambda mps,start: mps._merge_ABL_('R') if mps.cut==start else mps._merge_ABL_('L')
        poses={opt.site:mps1.table[opt.site] for opt in self}
        ms=sorted([opt.tensor for opt in self],key=lambda m:poses[m.labels[1]])
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
                result=contract([result,u1,ms[count],u2],engine='tensordot')
                count+=1
            else:
                news.remove(S1.prime)
                olds.remove(S1)
                u1.relabel(news=news,olds=olds)
                result=contract([result,u1,u2],engine='tensordot')
        if mps1 is mps2:
            mps1._set_ABL_(m,Lambda)
        else:
            mps1._set_ABL_(m1,Lambda1)
            mps2._set_ABL_(m2,Lambda2)
        return np.asarray(result)

    def to_mpo(self,degfres):
        '''
        Convert an optstr to the full-formated mpo.

        Parameters
        ----------
        degfres : DegFreTree
            The tree of the site degrees of freedom.

        Returns
        -------
        MPO
            The corresponding MPO.
        '''
        dtype,index=self[0].value.dtype,self[0].site.identifier
        type,layer=degfres[index].type if degfres.mode=='QN' else None,degfres.layers[degfres.level(index)-1]
        table,sites,bonds=degfres.table(layer),degfres.labels('S',layer),degfres.labels('O',layer)
        poses=set(table[opt.site.identifier] for opt in self)
        ms,count=[],0
        for pos in xrange(len(sites)):
            L,U,D,R=copy(bonds[pos]),sites[pos].prime,sites[pos],copy(bonds[pos+1])
            ndegfre=degfres.ndegfre(U.identifier)
            if degfres.mode=='QN':
                U,D=U.replace(qns=None),D.replace(qns=None)
                lqns,sqns=QuantumNumbers.mono(type.zero()) if pos==0 else ms[-1].labels[MPO.R].qns,sites[pos].qns
            if pos in poses:
                ms.append(Tensor((self[count].matrix*self[count].value).reshape((1,ndegfre,ndegfre,1)),labels=[L,U,D,R]))
                count+=1
            else:
                ms.append(Tensor(np.identity(sites[pos].dim,dtype=dtype).reshape((1,ndegfre,ndegfre,1)),labels=[L,U,D,R]))
            if degfres.mode=='QN':
                ms[-1].qng(axes=[MPO.L,MPO.U,MPO.D],qnses=[lqns,sqns,sqns],signs='++-')
        return MPO(ms)

    def connect(self,degfres):
        '''
        Connect the start and the end of an optstr by inserting identities in between them.

        Parameters
        ----------
        degfres : DegFreTree
            The tree of the site degrees of freedom.
        '''
        dtype=self[0].value.dtype
        layer=degfres.layers[degfres.level(self[0].site.identifier)-1]
        table,sites=degfres.table(layer),degfres.labels('S',layer)
        count,poses=0,[table[opt.site.identifier] for opt in self]
        for start,stop in zip(poses[:-1],poses[1:]):
            count+=1
            for pos in xrange(start+1,stop):
                self.insert(count,Opt.identity(sites[pos],dtype=dtype))
                count+=1

    def relayer(self,degfres,layer):
        '''
        Construt a new optstr with the site labels living on a specific layer of degfres.

        Parameters
        ----------
        degfres : DegFreTree
            The tree of the site degrees of freedom.
        layer : integer/tuple-of-string
            The layer where the site labels live.

        Returns
        -------
        OptStr
            The new optstr.
        '''
        new=layer if type(layer) in (int,long) else degfres.layers.index(layer)
        old=degfres.level(self[0].site.identifier)-1
        assert new>=0 and new<len(degfres.layers) and old>=new
        if old==new:
            return copy(self)
        else:
            poses={}
            for pos,opt in enumerate(self):
                index=opt.site.identifier
                ancestor=degfres.ancestor(index,generation=old-new)
                if ancestor in poses:
                    poses[ancestor][index]=pos
                else:
                    poses[ancestor]={index:pos}
            opts=[]
            olayer,nlayer=degfres.layers[old],degfres.layers[new]
            otable,ntable=degfres.table(olayer),degfres.table(nlayer)
            sites=degfres.labels('S',nlayer)
            for ancestor in sorted(poses.keys(),key=ntable.get):
                value,tag,m=1.0,(),1.0
                for index in degfres.descendants(ancestor,old-new):
                    if index in poses[ancestor]:
                        opt=self[poses[ancestor][index]]
                        value*=opt.value
                        tag+=(opt.tag,)
                        m=np.kron(m,opt.matrix)
                    else:
                        tag+=('i',)
                        m=np.kron(m,np.identity(degfres.ndegfre(index)))
                opts.append(Opt(value,sites[ntable[ancestor]],','.join(tag),m))
            return OptStr(opts)

class OptMPO(list):
    '''
    Matrix product operator in the form of single site operators.

    Attributes
    ----------
    sites : list of Label
        The site labels of the mpo.
    bonds : list of Label
        The bond labels of the mpo.
    '''

    def __init__(self,optstrs,degfres):
        '''
        Constructor.

        Parameters
        ----------
        optstrs : list of OptStr
            The optstrs contained in the mpo.
        degfres : DegFreTree
            The tree of the site degrees of freedom.
        '''
        dtype,layer=optstrs[0][0].value.dtype,degfres.layers[degfres.level(optstrs[0][0].site.identifier)-1]
        table,sites,bonds=degfres.table(layer),degfres.labels('S',layer),degfres.labels('O',layer)
        for optstr in optstrs: optstr.connect(degfres)
        optstrs.sort(key=lambda optstr: (len(optstr),table[optstr[0].site.identifier],tuple(opt.tag for opt in optstr)))
        rows,cols=[],[]
        for i,site in enumerate(sites):
            zero,identity=Opt.zero(site,dtype),Opt.identity(site,dtype)
            self.append(np.array([[zero,identity]] if i==0 else ([[identity],[zero]] if i==len(sites)-1 else [[identity,zero],[zero,identity]])))
            rows.append(1 if i==0 else 2)
            cols.append(1 if i==len(sites)-1 else 2)
        for optstr in optstrs:
            if len(optstr)==1:
                self[table[optstr[0].site.identifier]][-1,0]+=optstr[0]
            else:
                for i,opt in enumerate(optstr):
                    pos=table[opt.site.identifier]
                    if i==0:
                        col=[Opt.zero(opt.site,dtype)]*rows[pos]
                        col[-1]=opt
                        self[pos]=np.insert(self[pos],-1,col,axis=1)
                        cols[pos]+=1
                    elif i<len(optstr)-1:
                        row=[Opt.zero(opt.site,dtype)]*cols[pos]
                        self[pos]=np.insert(self[pos],-1,row,axis=0)
                        rows[pos]+=1
                        col=[Opt.zero(opt.site,dtype)]*rows[pos]
                        col[-2]=opt
                        self[pos]=np.insert(self[pos],-1,col,axis=1)
                        cols[pos]+=1
                    else:
                        row=[Opt.zero(opt.site,dtype)]*cols[pos]
                        row[0]=opt
                        self[pos]=np.insert(self[pos],-1,row,axis=0)
                        rows[pos]+=1
        self.sites=sites
        self.bonds=bonds

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        np.set_printoptions(formatter={'all': lambda element: element})
        for m in self:
            M,lengths=np.zeros(m.shape,dtype=object),np.zeros(m.shape,dtype=np.int64)
            for i,j in it.product(xrange(m.shape[0]),xrange(m.shape[1])):
                M[i,j]=repr(m[i,j])
                lengths[i,j]=len(M[i,j])
            lengths=lengths.max(axis=0)
            for i,j in it.product(xrange(m.shape[0]),xrange(m.shape[1])):
                M[i,j]=M[i,j].center(lengths[j])
            result.append(str(M))
        np.set_printoptions()
        return '\n'.join(result)

    def to_mpo(self,**karg):
        '''
        Convert to the tensor-formated mpo.

        Parameters
        ----------
        karg : dict with keys containing 'nsweep','method' and 'options'
            Please see MPO.compress for details.

        Returns
        -------
        MPO
            The corresponding tensor-formated MPO.
        '''
        Ms=[]
        dtype,type=self[0][0,0].value.dtype,self[0][0,0].site.qns.type if isinstance(self[0][0,0].site.qns,QuantumNumbers) else None
        for pos,m in enumerate(self):
            L,U,D,R,dim=copy(self.bonds[pos]),self.sites[pos].prime,self.sites[pos],copy(self.bonds[pos+1]),self.sites[pos].dim
            if type is not None: U,D=U.replace(qns=None),D.replace(qns=None)
            Ms.append(Tensor(np.zeros((m.shape[0],dim,dim,m.shape[1]),dtype=dtype),labels=[L,U,D,R]))
            for i,j in it.product(xrange(m.shape[0]),xrange(m.shape[1])):
                Ms[-1][i,:,:,j]=m[i,j].value*m[i,j].matrix
        result=MPO(Ms)
        result.compress(**karg)
        if type is not None:
            for pos in xrange(len(result)):
                lqns,sqns=QuantumNumbers.mono(type.zero()) if pos==0 else result[pos-1].labels[MPO.R].qns,self.sites[pos].qns
                result[pos].qng(axes=[MPO.L,MPO.U,MPO.D],qnses=[lqns,sqns,sqns],signs='++-')
        return result

class MPO(Arithmetic,list):
    '''
    Matrix product operator, with each of its elements a 4d `Tensor`.
    '''
    L,U,D,R=0,1,2,3

    def __init__(self,ms,sites=None,bonds=None):
        '''
        Constructor.

        Parameters
        ----------
        ms : list of 4d ndarray/Tensor
            The matrices of the mpo.
        sites : list of Label, optional
            The site labels of the mpo.
        bonds : list of Label, optional
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

    def relabel(self,sites,bonds):
        '''
        Change the labels of the mpo.

        Parameters
        ----------
        sites : list of Label
            The new site labels of the mpo.
        bonds : list of Label
            The new bond labels of the mpo.
        '''
        assert len(sites)==self.nsite and len(bonds)==self.nsite+1
        for m,L,S,R in zip(self,bonds[:-1],sites,bonds[1:]):
            m.relabel(news=[L,S.prime,S,R])

    def _mul_mpo_(self,other):
        '''
        The multiplication of two mpos.

        Parameters
        ----------
        other : MPO
            The other mpo.

        Returns
        -------
        MPO
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

        Parameters
        ----------
        other : MPS
            The mps.

        Returns
        -------
        MPS
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

    def __add__(self,other):
        '''
        Overloaded addition(+) operator, which supports the addition of two mpos.
        '''
        if isinstance(other,MPO):
            assert self.nsite==other.nsite
            ms=[]
            for i,(m1,m2) in enumerate(zip(self,other)):
                assert m1.labels==m2.labels
                labels=[label.replace(qns=None) for label in m1.labels]
                axes=[MPO.L,MPO.U,MPO.D] if i==0 else ([MPO.U,MPO.D,MPO.R] if i==self.nsite-1 else [MPO.U,MPO.D])
                ms.append(Tensor.directsum([m1,m2],labels=labels,axes=axes))
            return MPO(ms)
        else:
            assert abs(other)==0
            return self

    __iadd__=__add__

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

    def overlap(self,mps1,mps2):
        '''
        The overlap of an mpo between two mpses.

        Parameters
        ----------
        mps1,mps2 : MPS
            The two matrix product states between which the overlap of an mpo is calculated.

        Returns
        -------
        number
            The overlap.

        Notes
        -----
        Both mpses should be kets because in this function the complex conjugate of the first mps is always taken during the calculation.
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

        Parameters
        ----------
        nsweep : integer, optional
            The number of sweeps to compress the mpo.
        method : 'svd', 'dpl' or 'dln'
            The method used to compress the mpo.
        options : dict, optional
            The options used to compress the mpo.

        Returns
        -------
        MPO
            The compressed mpo.
        '''
        assert method in ('svd','dpl','dln')
        if method=='svd':
            tol=options.get('tol',TOL)
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

        Parameters
        ----------
        degfres : DegFreTree
            The tree of the site degrees of freedom.
        layer : integer/tuple-of-string
            The layer where the site labels live.
        nmax : integer, optional
            The maximum number of singular values to be kept.
        tol : np.float64, optional
            The tolerance of the singular values.

        Returns
        -------
        MPO
            The new mpo.
        '''
        new=layer if type(layer) in (int,long) else degfres.layers.index(layer)
        old=degfres.level(next(iter(self)).labels[MPO.U].identifier)-1
        assert new>=0 and new<len(degfres.layers)
        if new==old:
            return copy(self)
        else:
            olayer,nlayer=degfres.layers[old],degfres.layers[new]
            sites,bonds=degfres.labels('S',nlayer),degfres.labels('O',nlayer)
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
