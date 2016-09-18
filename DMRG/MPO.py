'''
Matrix product operator, including:
1) classes: OptStr
'''

__all__=['OptStr']

from numpy import *
from HamiltonianPy.Math.Tensor import *

class OptStr(list):
    '''
    '''
    
    def __init__(self,ms,labels):
        assert len(ms)==len(labels)
        for m,label in zip(ms,labels):
            assert ms.ndim==2
            assert isinstance(label,Label)
            self.append(Tensor(m,labels=[label.prime,label]))
        self.labels=set(labels)

    @staticmethod
    def from_operator(operator):
        pass

    def matrix(self,us,form='L'):
        table={u.labels[MPS.S]:i for i,u in enumerate(us)}
        if form=='L':
            start,count=table[self[0].labels[1]],0
            for i,u in enumerate(us[start:]):
                L,S,R=u.labels[MPS.L],u.labels[MPS.S],u.labels[MPS.R]
                up=u.copy(copy_data=False)
                if S in self.labels:
                    if i==0:
                        up.relabel(news=[S.prime,R.prime],olds=[S,R])
                        result=contract(up,self[count],u)
                    else:
                        up.relabel(news=[L.prime,S.prime,R.prime],olds=[L,S,R])
                        result=contract(result,up,self[count],u)
                    count+=1
                else:
                    up.relabel(news=[L.prime,R.prime],olds=[L,R])
                    result=contract(result,up,u)
        else:
            end,count=table[self[-1].labels[1]]+1,-1
            for i,u in enumerate(reversed(us[0:end])):
                L,S,R=u.labels[MPS.L],u.labels[MPS.S],u.labels[MPS.R]
                up=u.copy(copy_data=False)
                if S in self.labels:
                    if i==0:
                        up.relabel(news=[L.prime,S.prime],olds=[L,S])
                        result=contract(up,self[count],u)
                    else:
                        up.relabel(news=[L.prime,S.prime,R.prime],olds=[L,S,R])
                        result=contract(result,up,self[count],u)
                    count-=1
                else:
                    up.relabel(news=[L.prime,R.prime],olds=[L,R])
                    result=contract(result,up,u)
        return result

    def overlap(self,mps1,mps2):
        table={u.labels[MPS.S]:i for i,u in enumerate(mps1)}
        start,end,count=table[self[0].labels[1]],table[self[-1].labels[1]]+1,0
        if (mps1.cut<start and mps1.cut>end) or (mps2.cut<start and mps2.cut>end):
            raise ValueError("OptStr overlap error: both mps1.cut(%s) and mps2.cut(%s) should be in range [%s,%s]"%(mps1.cut,mps2.cut,start,end))
        def reset_and_protect(mps,start):
            if mps.cut==start:
                m,L=mps[mps.cut],mps.Lambda
                mps._reset_(merge='B',reset=None)
            else:
                m,L=mps[mps.cut-1],mps.Lambda
                mps._reset_(merge='A',reset=None)
            return m,L
        m1,L1=reset_and_protect(mps1,start)
        m2,L2=reset_and_protect(mps2,start)
        for i,(u1,u2) in enumerate(zip(mps1[start:end],mps2[start:end])):
            Lp,Sp,Rp=u1.labels[MPS.L],u1.labels[MPS.S],u1.labels[MPS.R]
            L,S,R=u2.labels[MPS.L],u2.labels[MPS.S],u2.labels[MPS.R]
            assert L==Lp and S==Sp and R==Rp
            if Sp in self.labels:
                news,olds=[Lp.prime,Sp.prime,Rp.prime],[Lp,Sp,Rp]
                if i==0:
                    news.remove(Lp.prime)
                    olds.remove(Lp)
                if i==end-start-1:
                    news.remove(Rp.prime)
                    olds.remove(Rp)
                u1.relabel(news=news,olds=olds)
                if i==0:
                    result=contract(u1,self[count],u2)
                else:
                    result=contract(result,u1,self[count],u2)
                count+=1
            else:
                u1.relabel(news=[Lp.prime,Rp.prime],olds=[Lp,Rp])
                result=contract(result,u1,u2)
        mps1._set_ABL_(m1,L1)
        mps2._set_ABL_(m2,L2)
        return result
