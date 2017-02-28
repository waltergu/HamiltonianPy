'''
Matrix product operator, including:
1) classes: OptStr, MPO
'''

__all__=['OptStr','MPO']

import numpy as np
from collections import OrderedDict
from HamiltonianPy import OperatorF,OperatorS,CREATION
from Tensor import Tensor,contract
from ..Misc import parity

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
            return OptStr(ms).relayer(degfres,layer)
        else:
            length=len(operator.indices)
            assert length%2==0
            table=degfres.table(degfres.layers[-1])
            permutation=sorted(range(length),key=lambda k:table[operator.indices[k].replace(nambu=None)])
            groups,counts=OrderedDict(),[]
            for k in permutation:
                leaf=table[operator.indices[k].replace(nambu=None)]
                m=np.array([[0.0,0.0],[1.0,0.0]]) if operator.indices[k].nambu==CREATION else np.array([[0.0,1.0],[0.0,0.0]])
                if pos in groups:
                    counts[-1]+=1
                    groups[leaf]=groups[leaf].dot(m)
                else:
                    counts.append(1)
                    groups[leaf]=m
            ms=[]
            keys=groups.keys()
            sites=degfres.labels(degfres.layers[-1],'S')
            zmatrix=np.array([[1.0,0.0],[0.0,-1.0]])
            for leaf in xrange(table[keys[0]],table[keys[-1]]+1):
                labels=[sites[leaf].prime,sites[leaf]]
                if leaf in groups:
                    assert counts[0] in (1,2)
                    length-=counts.pop(0)
                    ms.append(Tensor(groups[leaf] if length%2==0 else groups[leaf].dot(zmatrix),labels=labels))
                elif length%2!=0:
                    ms.append(Tensor(zmatrix,labels=labels))
            ms[0]=ms[0]*operator.value*parity(permutation)
            return OptStr(ms=ms).relayer(degfres,layer)

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
                The two matrix product state between which the overlap of an optstr is calculated.
                Note both mpses are kets, i.e. the complex conjugate of the inner product is taken in this function.
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
            if Sp in poses:
                u1.relabel(news=news,olds=olds)
                result=contract(result,u1,ms[count],u2,sequence='sequential')
                count+=1
            else:
                news.remove(S1.prime)
                olds.remove(S1)
                u1.relabel(news=news,olds=olds)
                result=contract(result,u1,u2,sequence='sequential')
        if mps1 is mps2:
            mps1._set_ABL_(m,Lambda)
        else:
            mps1._set_ABL_(m1,Lambda1)
            mps2._set_ABL_(m2,Lambda2)
        return np.asarray(result)

#    def relayer(self,degfres,layer,nmax=None,tol=None):
#        '''
#        Construt a new optstr with the site labels living on a specific layer of degfres.
#        Parameters:
#            degfres: DegFreTree
#                The tree of the site degrees of freedom.
#            layer: integer/tuple-of-string
#                The layer where the site labels live.
#            nmax: integer, optional
#                The maximum number of singular values to be kept.
#            tol: np.float64, optional
#                The tolerance of the singular values.
#        Returns: OptStr
#            The new optstr.
#        '''
#        pass

#    def to_mpo(self,degfres):
#        '''
#        Convert an optstr to the full-formated mpo.
#        Parameters:
#            degfres: DegFreTree
#                The tree of the site degrees of freedom.
#        Returns: MPO
#            The corresponding MPO.
#        '''
#        pass

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

#    def __add__(self,other):
#        '''
#        Overloaded addition(+) operator, which supports the addition of two mpos.
#        '''
#        pass

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
        self[0]*=other
        return self

    def __mul__(self,other):
        '''
        Overloaded multiplication(*) operator, which supports the multiplication of an mpo with a scalar.
        '''
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

#    def compress(self,nmax=None,tol=None):
#        '''
#        Compress the mpo.
#        Parameters:
#            nmax: integer, optional
#                The maximum number of singular values to be kept.
#            tol: float64, optional
#                The tolerance of the singular values.
#        Returns: MPO
#            The compressed mpo.
#        '''
#        pass

#    def relayer(self,degfres,layer,nmax=None,tol=None):
#        '''
#        Construt a new mpo with the site labels living on a specific layer of degfres.
#        Parameters:
#            degfres: DegFreTree
#                The tree of the site degrees of freedom.
#            layer: integer/tuple-of-string
#                The layer where the site labels live.
#            nmax: integer, optional
#                The maximum number of singular values to be kept.
#            tol: np.float64, optional
#                The tolerance of the singular values.
#        Returns: MPO
#            The new mpo.
#        '''
#        pass
