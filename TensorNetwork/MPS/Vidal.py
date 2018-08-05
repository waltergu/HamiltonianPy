'''
==========================
Vidal Matrix product state
==========================

Vidal matrix product states, including:
    * classes: Vidal
'''

__all__=['Vidal']

from MPS import MPS
from ..Tensor import *

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

    def __init__(self,Gammas=(),Lambdas=(),sites=None,bonds=None):
        '''
        Constructor.

        Parameters
        ----------
        Gammas : list of 3d ndarray/Tensor, optional
            The Gamma matrices on the site.
        Lambdas : list of 1d ndarray/Tensor, optional
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
            assert len(Gammas)==len(sites)==len(bonds)-1
            qnon=next(iter(sites)).qnon
            for Gamma,L,S,R in zip(Gammas,bonds[:-1],sites,bonds[1:]):
                assert Gamma.ndim==3
                if qnon:
                    L,S,R=L.replace(flow=+1),S.replace(flow=+1),R.replace(flow=-1)
                else:
                    L,S,R=L.replace(qns=Gamma.shape[Vidal.L],flow=0),S.replace(qns=Gamma.shape[Vidal.S],flow=0),R.replace(qns=Gamma.shape[Vidal.R],flow=0)
                self.Gammas.append(Tensor(Gamma,labels=[L,S,R]))
            for Lambda,label in zip(Lambdas,bonds[1:-1]):
                assert Lambda.ndim==1
                self.Lambdas.append(Tensor(Lambda,labels=[label if qnon else label.replace(qns=Lambda.shape[0])]))

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
                result=result*self.Lambdas[i-1]*Gamma
        return result.data.reshape((-1,))

    def tomixed(self,cut):
        '''
        Convert to the mixed MPS representation.

        Parameters
        ----------
        cut : int
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
                ms.append(Gamma if i==0 else self.Lambdas[i-1]*Gamma)
            else:
                ms.append(Gamma*self.Lambdas[i] if i<self.nsite-1 else Gamma)
        return MPS(ms=ms,Lambda=Lambda,cut=cut)
