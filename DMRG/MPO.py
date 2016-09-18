'''
Matrix product operator, including:
1) classes: OptStr
'''

__all__=['OptStr']

from numpy import *
from HamiltonianPy.Math.Tensor import *

class OptStr(object):
    '''
    '''
    
    def __init__(self,ms,labels):
        self.ms=[]
        assert len(ms)==len(labels)
        for m,label in zip(ms,labels):
            assert ms.ndim==2
            assert isinstance(label,Label)
            self.ms.append(Tensor(m,labels=[label.prime,label]))
        self.labels=set(labels)

    @staticmethod
    def from_operator(operator):
        pass

    def matrix(self,us,form='L'):
        table={u.labels[MPS.S]:i for i,u in enumerate(us)}
        if form=='L':
            start,count=table[self.ms[0].labels[1]],0
            for i,u in enumerate(us[start:]):
                L,S,R=u.labels[MPS.L],u.labels[MPS.S],u.labels[MPS.R]
                up=u.copy(copy_data=False)
                if S in self.labels:
                    if i==0:
                        up.relabel(news=[S.prime,R.prime],olds=[S,R])
                        result=contract(up,self.ms[count],u)
                    else:
                        up.relabel(news=[L.prime,S.prime,R.prime],olds=[L,S,R])
                        result=contract(result,up,self.ms[count],u)
                    count+=1
                else:
                    up.relabel(news=[L.prime,R.prime],olds=[L,R])
                    result=contract(result,up,u)
        else:
            end,count=table[self.ms[-1].labels[1]]+1,-1
            for i,u in enumerate(reversed(us[0:end])):
                L,S,R=u.labels[MPS.L],u.labels[MPS.S],u.labels[MPS.R]
                up=u.copy(copy_data=False)
                if S in self.labels:
                    if i==0:
                        up.relabel(news=[L.prime,S.prime],olds=[L,S])
                        result=contract(up,self.ms[count],u)
                    else:
                        up.relabel(news=[L.prime,S.prime,R.prime],olds=[L,S,R])
                        result=contract(result,up,self.ms[count],u)
                    count-=1
                else:
                    up.relabel(news=[L.prime,R.prime],olds=[L,R])
                    result=contract(result,up,u)
        return result

    def overlap(self,mps1,mps2):
        table={u.labels[MPS.S]:i for i,u in enumerate(us)}
        start,end,count=table[self.ms[0].labels[1]],table[self.ms[-1].labels[1]]+1,0
        
        
        for i,(u1,u2) in enumerate(zip(mps1.ms[start:end],mps2.ms[start:end])):
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
                    result=contract(u1,self.ms[count],u2)
                else:
                    result=contract(result,u1,self.ms[count],u2)
                count+=1
            else:
                u1.relabel(news=[Lp.prime,Rp.prime],olds=[Lp,Rp])
                result=contract(result,u1,u2)
        return result
