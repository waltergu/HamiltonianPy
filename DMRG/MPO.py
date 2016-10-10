'''
Matrix product operator, including:
1) classes: OptStr
'''

__all__=['OptStr']

from numpy import *
from HamiltonianPy.Basics import OperatorF,OperatorS
from HamiltonianPy.Math.Tensor import *
from MPS import *
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

    def __init__(self,value,ms,labels):
        '''
        Constructor.
        Parameters:
            value: number
                The overall coefficient of the optstr.
            ms: 2d ndarray
                The matrices of the mpo.
            labels: list of hashable objects
                The second physical labels of the mpo.
        NOTE: For each matrix in the mpo, the first physical label is just the prime of the second.
        '''
        assert len(ms)==len(labels)
        self.value=value
        for m,label in zip(ms,labels):
            assert m.ndim==2
            self.append(Tensor(m,labels=[prime(label),label]))
        self.labels=set(labels)

    @staticmethod
    def compose(value,ms):
        '''
        Constructor.
        Parameters:
            value: number
                The overall coefficient of the optstr.
            ms: list of 2d Tensor.
                The matrices of the mpo.
        Returns: OptStr
            The corresponding optstr.
        '''
        result,labels=OptStr.__new__(OptStr),[]
        result.value=value
        for m in ms:
            assert m.ndim==2
            assert m.labels[0]==prime(m.labels[1])
            result.append(m)
            labels.append(m.labels[1])
        result.labels=set(labels)
        return result

    @staticmethod
    def from_operator(operator,degfres,layer):
        '''
        Constructor, which convert an operator to its OptStr form.
        Parameters:
            operator: OperatorS, OperatorF
                The operator which is to be converted to the optstr form.
            degfres: DegFreTree
                The degfretree of the system.
            layer: string
                The layer where the optstr is to be transformed from the operator.
        Returns: OptStr
            The corresponding OptStr.
        '''
        if isinstance(operator,OperatorS):
            return opt_str_from_operator_s(operator,degfres,layer)
        elif isinstance(operator,OperatorF):
            return opt_str_from_operator_f(operator,degfres,layer)
        else:
            raise ValueError("OptStr.from_operator error: the class of the operator(%s) not supported."%(operator.__class__.__name__))

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return '\n'.join(str(m) for m in self)

    def matrix(self,us,form):
        '''
        The matrix representation of an optstr on a basis reprented by a mixed matrix product states.
        Parameters:
            us: MPS
                The representing mixed mps of the basis.
            form: 'L' or 'R'
                When 'L', us is left canonical;
                When 'R', us is right canonical.
        Returns: 2d Tensor
            The corresponding matrix representation of the optstr on the basis.
        '''
        result=Tensor(self.value,labels=[])
        temp=sorted(self,key=lambda m:us.table[m.labels[1]])
        if form=='L':
            start,count=us.table[temp[0].labels[1]],0
            for i,u in enumerate(us[start:]):
                L,S,R=u.labels[MPS.L],u.labels[MPS.S],u.labels[MPS.R]
                up=u.copy(copy_data=False).conjugate()
                if S in self.labels:
                    if i==0:
                        up.relabel(news=[prime(S),prime(R)],olds=[S,R])
                    else:
                        up.relabel(news=[prime(L),prime(S),prime(R)],olds=[L,S,R])
                    result=contract(result,up,temp[count],u)
                    count+=1
                else:
                    up.relabel(news=[prime(L),prime(R)],olds=[L,R])
                    result=contract(result,up,u)
        elif form=='R':
            end,count=us.table[temp[-1].labels[1]]+1,-1
            for i,u in enumerate(reversed(us[0:end])):
                L,S,R=u.labels[MPS.L],u.labels[MPS.S],u.labels[MPS.R]
                up=u.copy(copy_data=False).conjugate()
                if S in self.labels:
                    if i==0:
                        up.relabel(news=[prime(L),prime(S)],olds=[L,S])
                    else:
                        up.relabel(news=[prime(L),prime(S),prime(R)],olds=[L,S,R])
                    result=contract(result,up,temp[count],u)
                    count-=1
                else:
                    up.relabel(news=[prime(L),prime(R)],olds=[L,R])
                    result=contract(result,up,u)
        elif form==None:
            assert us.nsite==1 and len(self.labels)==1
            assert us[0] is None
            result=self[0]*result
        else:
            raise ValueError("OptStr matrix error: form(%s) not supported."%(form))
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
            news,olds=[prime(Lp),prime(Sp),prime(Rp)],[Lp,Sp,Rp]
            if i==0:
                news.remove(prime(Lp))
                olds.remove(Lp)
            if i==end-start-1:
                news.remove(prime(Rp))
                olds.remove(Rp)
            if Sp in self.labels:
                u1.relabel(news=news,olds=olds)
                result=contract(result,u1,temp[count],u2)
                count+=1
            else:
                news.remove(prime(Sp))
                olds.remove(Sp)
                u1.relabel(news=news,olds=olds)
                result=contract(result,u1,u2)
        if mps1 is mps2:
            mps1._set_ABL_(m,Lambda)
        else:
            mps1._set_ABL_(m1,Lambda1)
            mps2._set_ABL_(m2,Lambda2)
        return asarray(result)

def opt_str_from_operator_s(operator,degfres,layer):
    '''
    Convert an OperatorS to its corresponding OptStr form.
    For details, please see OptStr.from_operator.
    '''
    table=degfres.table(layer)
    branches=degfres.branches(layer)
    leaves=degfres.leaves(layer)
    qnc_merge_paths=degfres.qnc_merge_paths(layer)
    temp={}
    for i,index in enumerate(operator.indices):
        label=branches[index]
        if label in temp:
            temp[label][index]=i
        else:
            temp[label]={index:i}
    Ms=[]
    for label,seqs in temp.items():
        group=leaves[label]
        ms=[None]*len(group)
        for i,index in enumerate(group):
            if index in seqs:
                ms[i]=operator.spins[seqs[index]]
            else:
                ms[i]=identity(int(index.S*2)+1)
        M=1
        if qnc_merge_paths is None:
            for m in ms:
                M=kron(M,m)
        else:
            qncs=qnc_merge_paths[label]
            for m,qnc in zip(ms,qncs):
                M=kron(M,m)[qnc.permutation,:][:,qnc.permutation]
        Ms.append(Tensor(M,labels=[prime(label),label]))
    Ms.sort(key=lambda m:table[m.labels[1]])
    return OptStr.compose(operator.value,Ms)

def opt_str_from_operator_f(operator,degfres,layer):
    '''
    Convert an OperatorF to its corresponding OptStr form.
    For details, please see OptStr.from_operator.
    '''
    pass
