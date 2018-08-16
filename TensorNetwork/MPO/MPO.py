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
from HamiltonianPy import Arithmetic,RZERO,FockOperator,SOperator,JWBosonization,QuantumNumber,QuantumNumbers,decimaltostr
from HamiltonianPy.TensorNetwork.Tensor import *
from HamiltonianPy.TensorNetwork.MPS import MPS
from copy import copy

class Opt(Arithmetic):
    '''
    A single site operator.

    Attributes
    ----------
    site : Label
        The site label of the single site operator.
    content : dict in the form {tag:[coeff,matrix]}
        * tag : str
            The tag of a certain single site operator.
        * coeff : number
            The coefficient of a certain single site operator.
        * matrix : 2d ndarray
            The matrix of a certain single site operator.
    '''

    def __init__(self,site,content):
        '''
        Constructor.

        Parameters
        ----------
        site : Label
            The site label of the single site operator.
        content : dict in the form {tag:(coeff,matrix)}
            * tag : str
                The tag of a certain single site operator.
            * coeff : number
                The coefficient of a certain single site operator.
            * matrix : 2d ndarray
                The matrix of a certain single site operator.
        '''
        self.site=site
        self.content=content

    @property
    def matrix(self):
        '''
        The matrix of the single site operator.
        '''
        return np.sum(coeff*matrix for coeff,matrix in self.content.values())

    @property
    def tensor(self):
        '''
        The tensor representation of the single site operator.
        '''
        site=self.site.replace(flow=-1 if self.site.qnon else 0)
        return DTensor(self.matrix,labels=[site.P,site])

    @staticmethod
    def identity(site,dtype=np.float64):
        '''
        Construct an identity single site operator.

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
        return Opt(site,{'i':[1.0,np.identity(site.dim,dtype=dtype)]})

    @staticmethod
    def zero(site,dtype=np.float64):
        '''
        Construct a zero single site operator.

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
        return Opt(site,{'0':[1.0,np.zeros((site.dim,site.dim),dtype=dtype)]})

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return 'Opt(site=%s,content=%s)'%(self.site,self.content)

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return ''.join('%s%s%s'%('' if i==0 else '+' if coeff.real>0 else '-','' if coeff==1 else decimaltostr(coeff),tag) for i,(tag,(coeff,matrix)) in enumerate(self.content.items()))

    def __iadd__(self,other):
        '''
        Overloaded self addition(+) operator, which supports the self addition by an instance of Opt.
        '''
        assert other.site==self.site
        for tag,(coeff,matrix) in other.content.items():
            if tag in self.content:
                self.content[tag][0]+=coeff
                if norm(self.content[tag][0])<RZERO: self.content.pop(tag)
            else:
                if tag!='0': self.content[tag]=[coeff,matrix]
        if len(self.content)==0: self.content['0']=[1.0,np.zeros((self.site.dim,self.site.dim))]
        if len(self.content)>1: self.content.pop('0',None)
        return self

    def __add__(self,other):
        '''
        Overloaded left addition(+) operator, which supports the left addition by an instance of Opt.
        '''
        assert other.site==self.site
        result=Opt(self.site,{})
        result+=self
        result+=other
        return result

    def __imul__(self,other):
        '''
        Overloaded self-multiplication(*=) operator, which supports the self multiplication by an instance of Opt or a scalar.
        '''
        if isinstance(other,Opt):
            content={}
            for tag1,(coeff1,matrix1) in self.content.items():
                if tag1!='0' and norm(coeff1)>RZERO:
                    for tag2,(coeff2,matrix2) in self.content.items():
                        if tag2!='0' and norm (coeff2)>RZERO:
                            tag=tag2 if tag1=='i' else tag1 if tag2=='i' else '%s*%s'%(tag1,tag2)
                            if tag in content:
                                content[tag][0]+=coeff1*coeff2
                            else:
                                content[tag]=[coeff1*coeff2,matrix1.dot(matrix2)]
            self.content=content if len(content)>0 else {'0':[1.0,np.zeros((self.site.dim,self.site.dim))]}
        else:
            if norm(other)<RZERO:
                self.content.clear()
            else:
                for data in self.content.values(): data[0]*=other
        return self

    def __mul__(self,other):
        '''
        Overloaded left multiplication(*) operator, which supports the left multiplication by an instance of Opt or a scalar.
        '''
        content={}
        if isinstance(other,Opt):
            for tag1,(coeff1,matrix1) in self.content.items():
                if tag1!='0' and norm(coeff1)>RZERO:
                    for tag2,(coeff2,matrix2) in self.content.items():
                        if tag2!='0' and norm (coeff2)>RZERO:
                            tag=tag2 if tag1=='i' else tag1 if tag2=='i' else '%s*%s'%(tag1,tag2)
                            if tag in content:
                                content[tag][0]+=coeff1*coeff2
                            else:
                                content[tag]=[coeff1*coeff2,matrix1.dot(matrix2)]
        else:
            if norm(other)>=RZERO:
                for tag,(coeff,matrix) in self.content.items(): content[tag]=[coeff*other,matrix]
        return Opt(self.site,content if len(content)>0 else {'0':[1.0,np.zeros((self.site.dim,self.site.dim))]})

class OptStr(Arithmetic,list):
    '''
    Operator string, a list of single site operators.

    Attributes
    ----------
    dtype : np.float64, np.complex128, etc
        The data type of the operator string.
    '''

    def __init__(self,opts,dtype=np.float64):
        '''
        Constructor.

        Parameters
        ----------
        opts : list of Opt
            The single site operators of the optstr.
        dtype : np.float64, np.complex128, etc, optional
            The data type of the operator string.
        '''
        self.extend(opts)
        self.dtype=dtype

    @staticmethod
    def fromoperator(operator,degfres,layer=0):
        '''
        Constructor, which converts an operator to an optstr.

        Parameters
        ----------
        operator : SOperator/FockOperator
            The operator to be converted to an optstr.
        degfres : DegFreTree
            The degfretree of the system.
        layer : int/tuple-of-str, optional
            The layer where the converted optstr lives.

        Returns
        -------
        OptStr
            The corresponding OptStr.
        '''
        assert isinstance(operator,SOperator) or isinstance(operator,FockOperator)
        layer=degfres.layers[layer] if type(layer) is int else layer
        table,sites=degfres.table(degfres.layers[-1]),degfres.labels('S',degfres.layers[-1])
        operator=operator if isinstance(operator,SOperator) else JWBosonization(operator,table)
        opts=[]
        permutation=sorted(range(len(operator.indices)),key=lambda k: table[operator.indices[k]])
        for i,k in enumerate(permutation):
            index,matrix=operator.indices[k],operator.spins[k]
            opts.append(Opt(sites[table[index]],{matrix.tag:[operator.value if i==0 else 1.0,np.asarray(matrix)]}))
        return OptStr(opts,np.find_common_type([np.asarray(operator.value).dtype]+[matrix.dtype for matrix in operator.spins],[])).relayer(degfres,layer)

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return '\n'.join(str(opt) for opt in self)

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return ' '.join(repr(opt) for opt in self)

    def __imul__(self,other):
        '''
        Overloaded self-multiplication(*=) operator, which supports the self-multiplication by a scalar.
        '''
        self[0]*=other
        self.dtype=np.find_common_type([self.dtype,np.asarray(other).dtype],[])
        return self

    def __mul__(self,other):
        '''
        Overloaded multiplication(*) operator, which supports the multiplication of an optstr with a scalar.
        '''
        result=copy(self)
        result[0]=result[0]*other
        result.dtype=np.find_common_type([result.dtype,np.asarray(other).dtype],[])
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
        result=DTensor(1.0,labels=[])
        for i,(u1,u2) in enumerate(zip(mps1[start:stop],mps2[start:stop])):
            assert u1.labels==u2.labels
            u1,olds,news=u1.dagger,[],[]
            if i==0:
                olds.append(u1.labels[0])
                news.append(u1.labels[0].replace(prime=not u1.labels[0].prime))
            if i==stop-start-1:
                olds.append(u1.labels[2])
                news.append(u1.labels[2].replace(prime=not u1.labels[2].prime))
            if u1.labels[1].P in poses:
                u1.relabel(news=news,olds=olds)
                result=result*(u1,'force')*(ms[count],'force')*(u2,'force')
                count+=1
            else:
                olds.append(u1.labels[1])
                news.append(u1.labels[1].replace(prime=not u1.labels[1].prime))
                u1.relabel(news=news,olds=olds)
                result=result*(u1,'force')*(u2,'force')
        if mps1 is mps2:
            mps1._set_ABL_(m,Lambda)
        else:
            mps1._set_ABL_(m1,Lambda1)
            mps2._set_ABL_(m2,Lambda2)
        assert result.ndim==0
        return result.data

    def tompo(self,degfres,ttype='D'):
        '''
        Convert an optstr to the full-formatted mpo.

        Parameters
        ----------
        degfres : DegFreTree
            The tree of the site degrees of freedom.
        ttype : 'D'/'S', optional
            Tensor type. 'D' for dense and 'S' for sparse.

        Returns
        -------
        MPO
            The corresponding MPO.
        '''
        index=self[0].site.identifier
        type,layer=degfres.dtype,degfres.layers[degfres.level(index)-1]
        table,sites,bonds=degfres.table(layer),degfres.labels('S',layer),degfres.labels('O',layer)
        poses,matrices=set(table[opt.site.identifier] for opt in self),[opt.matrix for opt in self]
        ms,count=[],0
        for pos in range(len(sites)):
            ndegfre=sites[pos].dim
            if issubclass(type,QuantumNumbers):
                L=Label(bonds[pos],qns=QuantumNumbers.mono(type.zero()) if pos==0 else ms[-1].labels[MPO.R].qns,flow=None)
                U=sites[pos].P
                D=copy(sites[pos])
                R=Label(bonds[pos+1],qns=1,flow=None)
            else:
                L=Label(bonds[pos],qns=1,flow=0)
                U=sites[pos].P.replace(flow=0)
                D=sites[pos].replace(flow=0)
                R=Label(bonds[pos+1],qns=1,flow=0)
            if pos in poses:
                ms.append(DTensor(np.asarray(matrices[count],dtype=self.dtype).reshape((1,ndegfre,ndegfre,1)),labels=[L,U,D,R]))
                count+=1
            else:
                ms.append(DTensor(np.identity(ndegfre,dtype=self.dtype).reshape((1,ndegfre,ndegfre,1)),labels=[L,U,D,R]))
            if issubclass(type,QuantumNumbers): ms[-1].qngenerate(flow=-1,axes=[MPO.L,MPO.U,MPO.D],qnses=[L.qns,U.qns,D.qns],flows=[1,1,-1])
        if ttype=='S':
            assert issubclass(type,QuantumNumbers)
            for m in ms: m.qnsort()
            ms=[m.tostensor() for m in ms]
        return MPO(ms)

    def connect(self,degfres):
        '''
        Connect the start and the end of an optstr by inserting identities in between them.

        Parameters
        ----------
        degfres : DegFreTree
            The tree of the site degrees of freedom.
        '''
        layer=degfres.layers[degfres.level(self[0].site.identifier)-1]
        table,sites=degfres.table(layer),degfres.labels('S',layer)
        count,poses=0,[table[opt.site.identifier] for opt in self]
        for start,stop in zip(poses[:-1],poses[1:]):
            count+=1
            for pos in range(start+1,stop):
                self.insert(count,Opt.identity(sites[pos]))
                count+=1

    def relayer(self,degfres,layer):
        '''
        Construct a new optstr with the site labels living on a specific layer of degfres.

        Parameters
        ----------
        degfres : DegFreTree
            The tree of the site degrees of freedom.
        layer : int/tuple-of-str
            The layer where the site labels live.

        Returns
        -------
        OptStr
            The new optstr.
        '''
        new=layer if type(layer) is int else degfres.layers.index(layer)
        old=degfres.level(self[0].site.identifier)-1
        assert 0<=new<len(degfres.layers) and old>=new
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
            layer=degfres.layers[new]
            table=degfres.table(layer)
            sites=degfres.labels('S',layer)
            for ancestor in sorted(poses.keys(),key=table.get):
                tag,m=[],1.0
                for index in degfres.descendants(ancestor,old-new):
                    if index in poses[ancestor]:
                        opt=self[poses[ancestor][index]]
                        tag.append(repr(opt))
                        m=np.kron(m,opt.matrix)
                    else:
                        tag.append('i')
                        m=np.kron(m,np.identity(degfres.ndegfre(index)))
                opts.append(Opt(sites[table[ancestor]],{'(%s)'%('|'.join(tag)):[1.0,m]}))
            return OptStr(opts,self.dtype)

class OptMPO(list):
    '''
    Matrix product operator in the form of single site operators.

    Attributes
    ----------
    sites : list of Label
        The site labels of the mpo.
    bonds : list of str
        The bond identifiers of the mpo.
    dtype : np.float64, np.complex128, etc
        The data type of the matrix product operator.
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
        layer=degfres.layers[degfres.level(optstrs[0][0].site.identifier)-1]
        table,sites,bonds=degfres.table(layer),degfres.labels('S',layer),degfres.labels('O',layer)
        for optstr in optstrs: optstr.connect(degfres)
        optstrs.sort(key=lambda optstr: (len(optstr),table[optstr[0].site.identifier],tuple(repr(opt) for opt in optstr)))
        rows,cols=[],[]
        for i,site in enumerate(sites):
            self.append(    np.array([[Opt.zero(site),Opt.identity(site)]]) if i==0 else 
                            np.array([[Opt.identity(site)],[Opt.zero(site)]]) if i==len(sites)-1 else 
                            np.array([[Opt.identity(site),Opt.zero(site)],[Opt.zero(site),Opt.identity(site)]])
                            )
            rows.append(1 if i==0 else 2)
            cols.append(1 if i==len(sites)-1 else 2)
        for optstr in optstrs:
            if len(optstr)==1:
                self[table[optstr[0].site.identifier]][-1,0]+=optstr[0]
            else:
                for i,opt in enumerate(optstr):
                    pos=table[opt.site.identifier]
                    if i==0:
                        col=[Opt.zero(opt.site)]*rows[pos]
                        col[-1]=opt
                        self[pos]=np.insert(self[pos],-1,col,axis=1)
                        cols[pos]+=1
                    elif i<len(optstr)-1:
                        row=[Opt.zero(opt.site)]*cols[pos]
                        self[pos]=np.insert(self[pos],-1,row,axis=0)
                        rows[pos]+=1
                        col=[Opt.zero(opt.site)]*rows[pos]
                        col[-2]=opt
                        self[pos]=np.insert(self[pos],-1,col,axis=1)
                        cols[pos]+=1
                    else:
                        row=[Opt.zero(opt.site)]*cols[pos]
                        row[0]=opt
                        self[pos]=np.insert(self[pos],-1,row,axis=0)
                        rows[pos]+=1
        self.sites=sites
        self.bonds=bonds
        self.dtype=np.find_common_type([optstr.dtype for optstr in optstrs],[])

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        np.set_printoptions(formatter={'all': lambda element: element})
        for m in self:
            M,lengths=np.zeros(m.shape,dtype=object),np.zeros(m.shape,dtype=np.int64)
            for i,j in it.product(range(m.shape[0]),range(m.shape[1])):
                M[i,j]=repr(m[i,j])
                lengths[i,j]=len(M[i,j])
            lengths=lengths.max(axis=0)
            for i,j in it.product(range(m.shape[0]),range(m.shape[1])):
                M[i,j]=M[i,j].center(lengths[j])
            result.append(str(M))
        np.set_printoptions()
        return '\n'.join(result)

    def tompo(self,ttype='D',**karg):
        '''
        Convert to the tensor-formatted mpo.

        Parameters
        ----------
        ttype : 'D'/'S', optional
            Tensor type. 'D' for dense and 'S' for sparse.
        karg : dict with keys containing 'nsweep','method' and 'options'
            Please see MPO.compress for details.

        Returns
        -------
        MPO
            The corresponding tensor-formatted MPO.
        '''
        Ms=[]
        type=self[0][0,0].site.qns.type if isinstance(self[0][0,0].site.qns,QuantumNumbers) else None
        for pos,m in enumerate(self):
            dim=self.sites[pos].dim
            L=Label(self.bonds[pos],qns=m.shape[0],flow=0)
            U=self.sites[pos].replace(prime=True,qns=dim,flow=0)
            D=self.sites[pos].replace(qns=dim,flow=0)
            R=Label(self.bonds[pos+1],qns=m.shape[1],flow=0)
            Ms.append(DTensor(np.zeros((m.shape[0],dim,dim,m.shape[1]),dtype=self.dtype),labels=[L,U,D,R]))
            for i,j in it.product(range(m.shape[0]),range(m.shape[1])):
                Ms[-1].data[i,:,:,j]=m[i,j].matrix
        result=MPO(Ms)
        result.compress(**karg)
        if type is not None:
            for pos in range(len(result)):
                lqns,sqns=QuantumNumbers.mono(type.zero()) if pos==0 else result[pos-1].labels[MPO.R].qns,self.sites[pos].qns
                result[pos].qngenerate(flow=-1,axes=[MPO.L,MPO.U,MPO.D],qnses=[lqns,sqns,sqns],flows=[1,1,-1])
        if ttype=='S':
            assert issubclass(type,QuantumNumber)
            for i,m in enumerate(result):
                m.qnsort()
                result[i]=m.tostensor()
        return result

class MPO(Arithmetic,list):
    '''
    Matrix product operator, with each of its elements a 4d tensor.
    '''
    L,U,D,R=0,1,2,3

    def __init__(self,ms=(),ttype=None):
        '''
        Constructor.

        Parameters
        ----------
        ms : list of 4d DTensor/STensor, optional
            The matrices of the mpo.
        ttype : None/'D'/'S', optional
            Tensor type. 'D' for dense, 'S' for sparse and None for automatic.
        '''
        for m in ms:
            assert m.ndim==4 and m.labels[MPO.U]==m.labels[MPO.D].P
            self.append(Tensor(m,ttype=ttype))

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
        return [m.labels[MPO.D].replace(flow=None) for m in self]

    @property
    def bonds(self):
        '''
        The bond labels of the mpo.
        '''
        result=[]
        for i,m in enumerate(self):
            if i==0: result.append(m.labels[MPO.L].replace(flow=None))
            result.append(m.labels[MPO.R].replace(flow=None))
        return result

    @property
    def matrix(self):
        '''
        The normal matrix representation of the mpo.
        '''
        result=1.0
        ls,rs,L,R=[],[],1,1
        for m in self:
            ls.append(m.labels[MPO.U])
            rs.append(m.labels[MPO.D])
            L*=ls[-1].dim
            R*=rs[-1].dim
            result=result*m
        return result.transpose(axes=[self[0].labels[MPO.L]]+ls+rs+[self[-1].labels[MPO.R]]).toarray().reshape((L,R))

    @property
    def ttype(self):
        '''
        Tensor type of the mps.
        '''
        return self[0].ttype if len(self)>0 else None

    @staticmethod
    def compose(ms,sites,bonds,ttype=None):
        '''
        Constructor.

        Parameters
        ----------
        ms : list of 4d ndarray/dict
            The matrices of the mpo.
        sites : list of Label
            The site labels of the mpo.
        bonds : list of Label
            The bond labels of the mpo.
        ttype : None/'D'/'S', optional
            Tensor type. 'D' for dense, 'S' for sparse and None for automatic.
        '''
        assert len(ms)==len(sites)==len(bonds)-1
        result=MPO()
        qnon=sites[0].qnon
        for m,L,S,R in zip(ms,bonds[:-1],sites,bonds[1:]):
            assert m.ndim==4
            L=L.replace(flow=+1) if qnon else L.replace(qns=m.shape[0],flow=0)
            S=S.replace(flow=-1) if qnon else S.replace(qns=m.shape[1],flow=0)
            R=R.replace(flow=-1) if qnon else R.replace(qns=m.shape[3],flow=0)
            result.append(Tensor(m,labels=[L,S.P,S,R],ttype=ttype))
        return result

    @staticmethod
    def fromoperators(operators,degfres,layer=0,ttype='D'):
        '''
        Convert a collection of operators to matrix product operator.

        Parameters
        ----------
        operators : iterable of Operator
            The collection of operators.
        degfres : DegFreTree
            The degfretree of the system.
        layer : int/tuple-of-str, optional
            The layer where the converted mpo lives.
        ttype : 'D'/'S', optional
            Tensor type. 'D' for dense and 'S' for sparse.

        Returns
        -------
        MPO
            The converted matrix product operator.
        '''
        return OptMPO([OptStr.fromoperator(operator,degfres,layer) for operator in operators],degfres).tompo(ttype=ttype)

    def __getitem__(self,k):
        '''
        Operator "[]" for slicing.
        '''
        if isinstance(k,slice):
            assert k.step is None or k.step==1
            result=list.__new__(type(self))
            result.extend(list.__getitem__(self,k))
            return result
        else:
            return super(MPO,self).__getitem__(k)

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return '\n'.join('L: %s\nU: %s\nD: %s\nR: %s\ndata:\n%s'%(m.labels[MPO.L],m.labels[MPO.U],m.labels[MPO.D],m.labels[MPO.R],m.data) for m in self)

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return '\n'.join(str(m) for m in self)

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
                ms.append(directsum([m1,m2],labels=labels,axes=axes))
            return MPO(ms)
        else:
            assert abs(other)==0
            return self

    __iadd__=__add__

    def __imul__(self,other):
        '''
        Overloaded self-multiplication(*=) operator, which supports the self-multiplication by a scalar or an instance of MPO/MPS.
        '''
        if isinstance(other,MPO):
            return self._mul_mpo_(other)
        elif isinstance(other,MPS):
            return self._mul_mps_(other)
        else:
            self[0]*=other
            return self

    def __mul__(self,other):
        '''
        Overloaded multiplication(*) operator, which supports the left multiplication of an mpo by a scalar or an instance of MPO/MPS.
        '''
        if isinstance(other,MPO):
            result=self._mul_mpo_(other)
        elif isinstance(other,MPS):
            result=self._mul_mps_(other)
        else:
            result=copy(self)
            result[0]=result[0]*other
        return result

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
        for m1,m2 in zip(self,other):
            assert m1.labels==m2.labels and type(m1) is type(m2)
            m1,m2=copy(m1),copy(m2)
            L1,U1,D1,R1=m1.labels
            L2,U2,D2,R2=m2.labels
            if isinstance(m1,DTensor):
                L,linfo=Label.union([L1,L2],identifier=L1.identifier,flow=L1.flow,mode=0),None
                R,rinfo=Label.union([R1,R2],identifier=R1.identifier,flow=R1.flow,mode=0),None
            else:
                L,linfo=Label.union([L1,L2],identifier=L1.identifier,flow=L1.flow,mode=+2)
                R,rinfo=Label.union([R1,R2],identifier=R1.identifier,flow=R1.flow,mode=+2)
            s=Label('__MPO_MUL__',qns=U1.qns)
            l1,r1=L1.replace(identifier='__MPO_MUL_L1__'),R1.replace(identifier='__MPO_MUL_R1__')
            l2,r2=L2.replace(identifier='__MPO_MUL_L2__'),R2.replace(identifier='__MPO_MUL_R2__')
            m1.relabel(olds=[L1,D1,R1],news=[l1,s.replace(flow=D1.flow),r1])
            m2.relabel(olds=[L2,U2,R2],news=[l2,s.replace(flow=U2.flow),r2])
            ms.append((m1*m2).transpose((l1,l2,U1,D2,r1,r2)).merge(([l1,l2],L,linfo),([r1,r2],R,rinfo)))
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
        for m1,m2 in zip(self,other):
            assert type(m1) is type(m2)
            L1,U1,D1,R1=m1.labels
            L2,S,R2=m2.labels
            assert S==D1
            if isinstance(m1,DTensor):
                L,linfo=Label.union([L1,L2],identifier=L2.identifier,flow=L2.flow,mode=0),None
                R,rinfo=Label.union([R1,R2],identifier=R2.identifier,flow=R2.flow,mode=0),None
            else:
                L,linfo=Label.union([L1,L2],identifier=L2.identifier,flow=L2.flow,mode=+2)
                R,rinfo=Label.union([R1,R2],identifier=R2.identifier,flow=R2.flow,mode=+2)
            m=(m1*m2).transpose((L1,L2,U1,R1,R2)).merge(([L1,L2],L,linfo),([R1,R2],R,rinfo))
            m.relabel(olds=[U1],news=[S])
            ms.append(m)
        other._set_ABL_(u,Lambda)
        return MPS(ms=ms)

    def relabel(self,sites,bonds):
        '''
        Change the labels of the mpo.

        Parameters
        ----------
        sites : list of Label/str
            The new site labels of the mpo.
        bonds : list of Label/str
            The new bond labels of the mpo.
        '''
        assert len(sites)==self.nsite==len(bonds)-1
        fin,fot=(1,-1) if self[0].qnon else (0,0)
        for m,L,S,R in zip(self,bonds[:-1],sites,bonds[1:]):
            nl=L.replace(flow=fin) if isinstance(L,Label) else m.labels[MPO.L].replace(identifier=L)
            ns=S.replace(flow=fot) if isinstance(S,Label) else m.labels[MPO.D].replace(identifier=S)
            nr=R.replace(flow=fot) if isinstance(R,Label) else m.labels[MPO.R].replace(identifier=R)
            m.relabel(news=[nl,ns.P,ns,nr])

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
        assert self.nsite==mps1.nsite==mps2.nsite
        if mps1 is mps2:
            u,Lambda=mps1._merge_ABL_()
        else:
            u1,Lambda1=mps1._merge_ABL_()
            u2,Lambda2=mps2._merge_ABL_()
        result=DTensor(1.0,labels=[])
        for i,(mpo,m1,m2) in enumerate(zip(self,mps1,mps2)):
            m1,olds,news=m1.dagger,[],[]
            if i==0:
                olds.append(m1.labels[0])
                news.append(m1.labels[0].replace(prime=not m1.labels[0].prime))
            if i==self.nsite-1:
                olds.append(m1.labels[2])
                news.append(m1.labels[2].replace(prime=not m1.labels[2].prime))
            m1.relabel(olds=olds,news=news)
            result=result*m1*mpo*m2
        if mps1 is mps2:
            mps1._set_ABL_(u,Lambda)
        else:
            mps1._set_ABL_(u1,Lambda1)
            mps2._set_ABL_(u2,Lambda2)
        assert result.shape==(1,1)
        return result.toarray()[0,0]

    def tosparse(self):
        '''
        Convert dense tensors to sparse tensors.
        '''
        return MPO(ms=(m.tostensor() for m in self))

    def todense(self):
        '''
        Convert sparse tensors to dense tensors.
        '''
        return MPO(ms=(m.todtensor() for m in self))

    def compress(self,nsweep=1,method='dpl',options=None):
        '''
        Compress the mpo.

        Parameters
        ----------
        nsweep : int, optional
            The number of sweeps to compress the mpo.
        method : 'svd'/'dpl', optional
            The method used to compress the mpo.
        options : dict, optional
            The options used to compress the mpo.

        Returns
        -------
        MPO
            The compressed mpo.
        '''
        assert method in ('svd','dpl')
        options=options or {}
        if method=='svd':
            tol=options.get('tol',5*RZERO)
            for _ in range(nsweep):
                for i,m in enumerate(self):
                    if i<self.nsite-1:
                        L,U,D,R=m.labels
                        u,s,v=svd(m,row=[L,U,D],new=Label('__MPO_compress_svd__',None,None),col=[R],tol=tol)
                        self[i+0]=u*s
                        self[i+1]=v*self[i+1]
                        self[i+0].relabel(olds=[MPO.R],news=[self[i+0].labels[MPO.R].replace(identifier=R.identifier)])
                        self[i+1].relabel(olds=[MPO.L],news=[self[i+1].labels[MPO.L].replace(identifier=R.identifier)])
                for i,m in enumerate(reversed(self)):
                    if i<self.nsite-1:
                        L,U,D,R=m.labels
                        u,s,v=svd(m,row=[L],new=Label('__MPO_compress_svd__',None,None),col=[U,D,R],tol=tol)
                        self[-1-i]=s*v
                        self[-2-i]=self[-2-i]*u
                        self[-1-i].relabel(olds=[MPO.L],news=[self[-1-i].labels[MPO.L].replace(identifier=L.identifier)])
                        self[-2-i].relabel(olds=[MPO.R],news=[self[-2-i].labels[MPO.R].replace(identifier=L.identifier)])
                for m in self:
                    m[np.abs(m)<tol]=0.0
        else:
            assert all(isinstance(m,DTensor) for m in self)
            zero,tol=options.get('zero',10**-8),options.get('tol',10**-6)
            for _ in range(nsweep):
                for i,m in enumerate(reversed(self)):
                    if i<self.nsite-1:
                        L,U,D,R=m.labels
                        T,M=deparallelization(m,row=[L],new=Label('__MPO_compress_dpl__',None,None),col=[U,D,R],mode='R',zero=zero,tol=tol)
                        self[-1-i]=M
                        self[-2-i]=self[-2-i]*T
                        self[-1-i].relabel(olds=[MPO.L],news=[self[-1-i].labels[MPO.L].replace(identifier=L.identifier)])
                        self[-2-i].relabel(olds=[MPO.R],news=[self[-2-i].labels[MPO.R].replace(identifier=L.identifier)])
                for i,m in enumerate(self):
                    if i<self.nsite-1:
                        L,U,D,R=m.labels
                        M,T=deparallelization(m,row=[L,U,D],new=Label('__MPO_compress_dpl__',None,None),col=[R],mode='C',zero=zero,tol=tol)
                        self[i+0]=M
                        self[i+1]=T*self[i+1]
                        self[i+0].relabel(olds=[MPO.R],news=[self[i+0].labels[MPO.R].replace(identifier=R.identifier)])
                        self[i+1].relabel(olds=[MPO.L],news=[self[i+1].labels[MPO.L].replace(identifier=R.identifier)])

    def impoprediction(self,sites,bonds,ttype=None):
        '''
        Infinite mpo prediction.

        Parameters
        ----------
        sites,bonds : list of Label/str
            The site/bond labels/identifiers of the new mpo.
        ttype : None/'D'/'S', optional
            Tensor type. 'D' for dense, 'S' for sparse and None for automatic.

        Returns
        -------
        MPO
            The predicted mpo.
        '''
        assert self.nsite==len(sites)==len(bonds)-1 and self.nsite%2==0 
        assert self[0].labels[MPO.L].qns==self[self.nsite//2].labels[MPO.L].qns==self[-1].labels[MPO.R].qns
        ms=[]
        for m,L,S,R in zip(self,bonds[:-1],sites,bonds[1:]):
            L=m.labels[MPO.L].replace(identifier=L.identifier if isinstance(L,Label) else L)
            U=m.labels[MPO.U].replace(identifier=S.identifier if isinstance(S,Label) else S)
            D=m.labels[MPO.D].replace(identifier=S.identifier if isinstance(S,Label) else S)
            R=m.labels[MPO.R].replace(identifier=R.identifier if isinstance(R,Label) else R)
            ms.append(Tensor(m.data,labels=[L,U,D,R]))
        return MPO(ms,ttype=ttype)

    def impogrowth(self,sites,bonds,ttype=None):
        '''
        Infinite mpo growth.

        Parameters
        ----------
        sites,bonds : list of Label/str
            The site/bond labels/identifiers of the new mpo.
        ttype : None/'D'/'S', optional
            Tensor type. 'D' for dense, 'S' for sparse and None for automatic.

        Returns
        -------
        MPO
            The mpo after growth.
        '''
        assert self.nsite>0 and self.nsite%2==0 and len(sites)+1==len(bonds)
        ob,nb=self.nsite//2+1,(len(bonds)+1)//2
        cms=self[2*ob-nb-1:nb-1].impoprediction(sites[ob-1:-ob+1],bonds[ob-1:-ob+1])
        lms=self[:ob-1]
        rms=self[ob-1:]
        lms.relabel(sites[:ob-1],bonds[:ob])
        rms.relabel(sites[-ob+1:],bonds[-ob:])
        return MPO(it.chain(lms,cms,rms),ttype=ttype)

    def relayer(self,degfres,layer,nmax=None,tol=None):
        '''
        Construct a new mpo with the site labels living on a specific layer of degfres.

        Parameters
        ----------
        degfres : DegFreTree
            The tree of the site degrees of freedom.
        layer : int/tuple-of-str
            The layer where the site labels live.
        nmax : int, optional
            The maximum number of singular values to be kept.
        tol : np.float64, optional
            The tolerance of the singular values.

        Returns
        -------
        MPO
            The new mpo.
        '''
        assert all(isinstance(m,DTensor) for m in self)
        new=layer if type(layer) is int else degfres.layers.index(layer)
        old=degfres.level(next(iter(self)).labels[MPO.U].identifier)-1
        assert 0<=new<len(degfres.layers)
        if new==old:
            return copy(self)
        else:
            olayer,nlayer=degfres.layers[old],degfres.layers[new]
            sites,bonds=[site.replace(flow=-1 if site.qnon else 0) for site in degfres.labels('S',nlayer)],degfres.labels('O',nlayer)
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
                    M=np.product(ms)
                    o1,o2=M.labels[0],M.labels[-1]
                    n1,n2=o1.replace(identifier=bonds[i]),o2.replace(identifier=bonds[i+1])
                    M.relabel(olds=[o1,o2],news=[n1,n2])
                    Ms.append(M.transpose([n1]+ups+dws+[n2]).merge((ups,site.P),(dws,site)))
            else:
                table,s,v=degfres.table(nlayer),1.0,1.0
                for i,m in enumerate(self):
                    m=s*v*m
                    L,U,D,R=m.labels
                    indices=degfres.descendants(D.identifier,generation=new-old)
                    start,stop=table[indices[0]],table[indices[-1]]+1
                    ups,dws,orders,labels,qnses=[],[],[],[],[]
                    for j,site in enumerate(sites[start:stop]):
                        ups.append(site.P)
                        dws.append(site)
                        orders.append(ups[-1])
                        orders.append(dws[-1])
                        qns=QuantumNumbers.kron([site.qns]*2,signs=(+1,-1)) if site.qnon else site.dim**2
                        labels.append(Label('__MPO_RELAYER_%s__'%j,qns=qns,flow=+1 if site.qnon else 0))
                        qnses.append(qns)
                    S=Label('__MPO_RELAYER__',qns=(QuantumNumbers.kron if U.qnon else np.product)(qnses),flow=+1 if U.qnon else 0)
                    m=m.split((U,ups),(D,dws)).transpose([L]+orders+[R]).merge((orders,S))
                    us,s,v=expandedsvd(m,L=[L],S=S,R=[R],E=labels,I=[Label(bond,None,None) for bond in bonds[start+1:stop+1]],cut=stop-start,nmax=nmax,tol=tol)
                    for u,up,dw,label in zip(us,ups,dws,labels):
                        Ms.append(u.split((label,[up,dw])))
                Ms[-1]=Ms[-1]*s*v
                Ms[+0].relabel(olds=[MPO.L],news=[Ms[+0].labels[MPO.L].replace(identifier=bonds[+0])])
                Ms[-1].relabel(olds=[MPO.R],news=[Ms[-1].labels[MPO.R].replace(identifier=bonds[-1])])
            return MPO(Ms)
