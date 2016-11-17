'''
Matrix product operator, including:
1) classes: OptStr
'''

__all__=['OptStr']

from numpy import *
from HamiltonianPy.Basics import OperatorF,OperatorS,CREATION
from HamiltonianPy.Math.Tensor import Tensor,contract
from HamiltonianPy.Math.linalg import parity
from MPS import *
from collections import OrderedDict
import warnings

class OptStr(list):
    '''
    Operator string, which is a special kind of matrix product operator, with the internal indices of the matrices trivial.
        For each of its elements: 2d Tensor
            The matrices of the mpo.
    Attribues:
        value: number
            The overall coefficient of the optstr.
        labels: set of hashable objects
            The physical labels of the mpo.
    '''

    def __init__(self,value,ms,labels=None):
        '''
        Constructor.
        Parameters:
            value: number
                The overall coefficient of the optstr.
            ms: 2d ndarray/Tensor
                The matrices of the mpo.
            labels: list of hashable objects, optional
                The second physical labels of the mpo.
        NOTE: For each matrix in the mpo, the first physical label is just the prime of the second.
        '''
        self.value=value
        self.labels=set()
        if labels is None:
            for m in ms:
                self.append(m)
        else:
            assert len(ms)==len(labels)
            for m,label in zip(ms,labels):
                self.append(Tensor(m,labels=[label.prime,label]))

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        result.append('value: %s'%self.value)
        result.extend([str(m) for m in self])
        return '\n'.join(result)

    def append(self,m):
        '''
        Overloaded append.
        '''
        assert isinstance(m,Tensor) and m.ndim==2 and m.labels[0]==m.labels[1].prime
        list.append(self,m)
        self.labels.add(m.labels[1])

    def insert(self,index,m):
        '''
        Overloaded insert.
        '''
        assert isinstance(m,Tensor) and m.ndim==2 and m.labels[0]==m.labels[1].prime
        list.insert(self,index,m)
        self.labels.add(m.labels[1])

    def extend(self,ms):
        '''
        Overloaded extend.
        '''
        for m in ms:
            self.append(m)

    @staticmethod
    def from_operator(operator,degfres,layer):
        '''
        Constructor, which convert an operator to its OptStr form.
        Parameters:
            operator: OperatorS, OperatorF
                The operator which is to be converted to the optstr form.
            degfres: DegFreTree
                The degfretree of the system.
            layer: tuple of string
                The layer where the optstr is to be transformed from the operator.
        Returns: OptStr
            The corresponding OptStr.
        '''
        if isinstance(operator,OperatorS):
            return optstr_from_operators(operator,degfres,layer)
        elif isinstance(operator,OperatorF):
            return optstr_from_operatorf(operator,degfres,layer)
        else:
            raise ValueError("OptStr.from_operator error: the class of the operator(%s) not supported."%(operator.__class__.__name__))

    def split(self,A,B,coeff=None):
        '''
        Split the optstr according to A and B.
        Parameters:
            A/B: set/dict/list of Index
                The labels of A/B part.
            coeff: None, 'A' or 'B'
                Plase see below.
        Returns:
            1) When coeff is None, the returns is a 3-tuple in the form (value,optstr_a,optstr_b)
                value: number
                    The coefficient of the optstr.
                optstr_a/optstr_b: OptStr
                    The A/B part of the optstr.
            2) When coeff is 'A' or 'B', the returns is a 2-tuple in the form (optstr_a,optstr_b)
                optstr_a/optstr_b: OptStr
                    The A/B part of the optstr.
                    If coeff=='A', the coefficient of the optstr will be absorbed into optstr_a;
                    If coeff=='B', the coefficient of the optstr will be absorbed into optstr_b.
        '''
        assert coeff in ('A','B',None)
        if coeff is None:
            value=self.value
            optstr_a=OptStr(1.0,[m for m in self if m.labels[1] in A])
            optstr_b=OptStr(1.0,[m for m in self if m.labels[1] in B])
            return value,optstr_a,optstr_b
        else:
            if coeff=='A':
                optstr_a=OptStr(self.value,[m for m in self if m.labels[1] in A])
                optstr_b=OptStr(1.0,[m for m in self if m.labels[1] in B])
                return optstr_a,optstr_b
            else:
                optstr_a=OptStr(1.0,[m for m in self if m.labels[1] in A])
                optstr_b=OptStr(self.value,[m for m in self if m.labels[1] in B])
                return optstr_a,optstr_b

    def matrix(self,us,form):
        '''
        The matrix representation of an optstr on a basis reprented by a mixed matrix product states.
        Parameters:
            us: MPS
                The representing mixed mps of the basis.
            form: 'L', 'R' or 'S'
                When 'L', us is left canonical;
                When 'R', us is right canonical;
                When 'S', us is omitted.
        Returns: 2d Tensor
            The corresponding matrix representation of the optstr on the basis.
        '''
        result=Tensor(self.value,labels=[])
        if form=='S' or us.nsite==0:
            assert len(self)==1
            result=self[0]*result
        else:
            temp=sorted(self,key=lambda m:us.table[m.labels[1]])
            if form=='L':
                start,count=us.table[temp[0].labels[1]],0
                for i,u in enumerate(us[start:]):
                    L,S,R=u.labels[MPS.L],u.labels[MPS.S],u.labels[MPS.R]
                    up=u.copy(copy_data=False).conjugate()
                    if S in self.labels:
                        if i==0:
                            up.relabel(news=[S.prime,R.prime],olds=[S,R])
                        else:
                            up.relabel(news=[L.prime,S.prime,R.prime],olds=[L,S,R])
                        result=contract(result,up,temp[count],u)
                        count+=1
                    else:
                        up.relabel(news=[L.prime,R.prime],olds=[L,R])
                        result=contract(result,up,u)
            elif form=='R':
                end,count=us.table[temp[-1].labels[1]]+1,-1
                for i,u in enumerate(reversed(us[0:end])):
                    L,S,R=u.labels[MPS.L],u.labels[MPS.S],u.labels[MPS.R]
                    up=u.copy(copy_data=False).conjugate()
                    if S in self.labels:
                        if i==0:
                            up.relabel(news=[L.prime,S.prime],olds=[L,S])
                        else:
                            up.relabel(news=[L.prime,S.prime,R.prime],olds=[L,S,R])
                        result=contract(result,up,temp[count],u)
                        count-=1
                    else:
                        up.relabel(news=[L.prime,R.prime],olds=[L,R])
                        result=contract(result,up,u)
            else:
                raise ValueError("OptStr matrix error: form(%s) not supported."%(form))
            if result.labels[1]._prime_:
                result=result.transpose(axes=[1,0])
        return result

    def overlap(self,mps1,mps2):
        '''
        The overlap of an optstr between two mpses.
        Parameters:
            mps1,mps2: MPS
                The two matrix product state between which the overlap of an optstr is calculated.
                Note both mpses are kets, i.e. the complex conjugate of the inner product is taken in this function.
        Returns: number
            The overlap.
        '''
        def reset_and_protect(mps,start):
            if mps.cut==start:
                m,Lambda=mps[mps.cut],mps.Lambda
                mps._reset_(merge='R',reset=None)
            else:
                m,Lambda=mps[mps.cut-1],mps.Lambda
                mps._reset_(merge='L',reset=None)
            return m,Lambda
        assert mps1.table==mps2.table
        temp=sorted(self,key=lambda m:mps1.table[m.labels[1]])
        if mps1 is mps2:
            start,end,count=mps1.table[temp[0].labels[1]],mps1.table[temp[-1].labels[1]]+1,0
            if mps1.cut<start or mps1.cut>end:
                warnings.warn("OptStr overlap warning: the cut of the mps is %s and will be moved into the range [%s,%s]."%(mps1.cut,start,end))
                if mps1.cut<start:
                    mps1>>=start-mps1.cut
                else:
                    mps1<<=mps1.cut-end
            m,Lambda=reset_and_protect(mps1,start)
        else:
            start,end,count=0,mps1.nsite,0
            m1,Lambda1=reset_and_protect(mps1,start)
            m2,Lambda2=reset_and_protect(mps2,start)
        result=Tensor(self.value,labels=[])
        for i,(u1,u2) in enumerate(zip(mps1[start:end],mps2[start:end])):
            u1=u1.copy(copy_data=False).conjugate()
            Lp,Sp,Rp=u1.labels[MPS.L],u1.labels[MPS.S],u1.labels[MPS.R]
            L,S,R=u2.labels[MPS.L],u2.labels[MPS.S],u2.labels[MPS.R]
            assert L==Lp and S==Sp and R==Rp
            news,olds=[Lp.prime,Sp.prime,Rp.prime],[Lp,Sp,Rp]
            if i==0:
                news.remove(Lp.prime)
                olds.remove(Lp)
            if i==end-start-1:
                news.remove(Rp.prime)
                olds.remove(Rp)
            if Sp in self.labels:
                u1.relabel(news=news,olds=olds)
                result=contract(result,u1,temp[count],u2)
                count+=1
            else:
                news.remove(Sp.prime)
                olds.remove(Sp)
                u1.relabel(news=news,olds=olds)
                result=contract(result,u1,u2)
        if mps1 is mps2:
            mps1._set_ABL_(m,Lambda)
        else:
            mps1._set_ABL_(m1,Lambda1)
            mps2._set_ABL_(m2,Lambda2)
        return asarray(result)

    @staticmethod
    def from_contents(value,contents,degfres,layer):
        '''
        Construt an optstr from contents by combining "small" physical degrees of freedom into "big" ones.
        Parameters:
            value: number
                The coefficient of the optstr.
            contents: list 2-tuple in the form (leaf,m)
                leaf: Index
                    The original "small" physical degrees of freedom.
                m: 2d ndarray
                    The matrix corresponding to the "small" physical degrees of freedom.
            degfres: DegFreTree
                The physical degrees of freedom.
            layer: tuple of string
                The layer where the "big" physical degrees of freedom of the optstr are restricted.
        Returns: OptStr
            The constructed optstr.
        '''
        ms,temp=[],{}
        for i,(leaf,m) in enumerate(contents):
            assert degfres.is_leaf(leaf)
            branch=degfres.branches(layer)[leaf]
            ms.append(m)
            if branch in temp:
                temp[branch][leaf]=i
            else:
                temp[branch]={leaf:i}
        Ms=[1]*len(temp)
        for i,(branch,seqs) in enumerate(temp.items()):
            buff=[(ms[seqs[index]] if index in seqs else identity(degfres.ndegfre(index))) for index in degfres.leaves(layer)[branch]]
            if degfres.mode=='NB':
                for m in buff:
                    Ms[i]=kron(Ms[i],m)
            else:
                for m,qnc in zip(buff,degfres.qnc_kron_paths(layer)[branch]):
                    Ms[i]=qnc.reorder(kron(Ms[i],m),axes=[0,1])
            label=degfres.labels(layer,full_labels=False)[branch]
            Ms[i]=Tensor(Ms[i],labels=[label.prime,label])
        Ms.sort(key=lambda m:degfres.table(layer)[m.labels[0].identifier])
        return OptStr(value,Ms)

def optstr_from_operators(operator,degfres,layer):
    '''
    Convert an OperatorS to its corresponding OptStr form.
    For details, please see OptStr.from_operator.
    '''
    return OptStr.from_contents(operator.value,[(index,matrix) for index,matrix in zip(operator.indices,operator.spins)],degfres,layer)

def optstr_from_operatorf(operator,degfres,layer):
    '''
    Convert an OperatorF to its corresponding OptStr form.
    For details, please see OptStr.from_operator.
    '''
    length=len(operator.indices)
    assert length%2==0
    table=degfres.table(degfres.layers[-1])
    permutation=sorted(range(length),key=lambda k:table[operator.indices[k].replace(nambu=None)])
    temp,counts=OrderedDict(),[]
    for k in permutation:
        leaf=operator.indices[k].replace(nambu=None)
        m=array([[0.0,1.0],[0.0,0.0]]) if operator.indices[k].nambu==CREATION else array([[0.0,0.0],[1.0,0.0]])
        if leaf in temp:
            counts[-1]+=1
            temp[leaf]=temp[leaf].dot(m)
        else:
            counts.append(1)
            temp[leaf]=m
    contents=[]
    keys,reversed_table=temp.keys(),degfres.reversed_table(degfres.layers[-1])
    for i in xrange(table[keys[0]],table[keys[-1]]+1):
        leaf=reversed_table[i]
        if leaf in temp:
            assert counts[0] in (1,2)
            length-=counts.pop(0)
            contents.append((leaf,temp[leaf] if length%2==0 else temp[leaf].dot(array([[-1.0,0.0],[0.0,1.0]]))))
        elif length%2!=0:
            contents.append((leaf,array([[-1.0,0.0],[0.0,1.0]])))
    return OptStr.from_contents(operator.value*parity(permutation),contents,degfres,layer)
