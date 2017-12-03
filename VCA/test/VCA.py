'''
VCA test.
'''

__all__=['test_vca']

from HamiltonianPy import *
import HamiltonianPy.ED as ED
import HamiltonianPy.VCA as VCA
import numpy as np

def test_vca():
    print 'test_vca'
    t,U,m,n=-1.0,8.0,2,2
    basis=FBasis(2*m*n,m*n,0.0)
    cell=Square('S1')('1P-1P',1)
    lattice=Square('S1')('%sP-%sP'%(m,n),1)
    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=lambda pid: Fermi(atom=0,norbital=1,nspin=2,nnambu=1))
    cgf=ED.FGF(operators=fspoperators(config.table(),lattice),nstep=200,prepare=ED.EDGFP,savedata=False,run=ED.EDGF)
    vca=VCA.VCA(
            name=       'WG-%s-%s'%(lattice.name,basis.rep),
            cgf=        cgf,
            sectors=    [basis],
            cell=       cell,
            lattice=    lattice,
            config=     config,
            terms=      [Hopping('t',t,neighbour=1),Hubbard('U',U)],
            weiss=      [Onsite('afm',0.0,indexpacks=sigmaz('sp'),amplitude=lambda bond: 1 if bond.spoint.pid.site in (0,3) else -1,modulate=True)],
            mask=       ['nambu'],
            dtype=      np.float64
            )
    vca.add(GP(name='GP',mu=U/2,BZ=square_bz(reciprocals=lattice.reciprocals,nk=100),run=VCA.VCAGP))
    vca.register(VCA.GPM(name='afm-1',BS=BaseSpace(('afm',np.linspace(0.0,0.3,16))),dependences=['GP'],savedata=False,run=VCA.VCAGPM))
    vca.register(VCA.GPM(name='afm-2',BS={'afm':0.1},options={'method':'BFGS','options':{'disp':True}},dependences=['GP'],savedata=False,run=VCA.VCAGPM))
    vca.register(VCA.EB(name='EB',parameters={'afm':0.20},path=square_gxm(nk=100),mu=U/2,emax=6.0,emin=-6.0,eta=0.05,ne=400,savedata=False,run=VCA.VCAEB))
    vca.register(DOS(name='DOS',parameters={'afm':0.20},BZ=KSpace(reciprocals=lattice.reciprocals,nk=20),mu=U/2,emin=-10,emax=10,ne=400,eta=0.05,savedata=False,run=VCA.VCADOS))
    vca.register(FS(name='FS',parameters={'afm':0.20},mu=U/2,BZ=square_bz(nk=100),savedata=False,run=VCA.VCAFS))
    vca.register(VCA.OP(name='OP',parameters={'afm':0.20},mu=U/2,terms=vca.weiss,BZ=square_bz(reciprocals=lattice.reciprocals,nk=100),run=VCA.VCAOP))
    vca.register(VCA.DTBT(name='DTBT',parameters={'afm':0.20},path=square_gxm(nk=100),mu=U/2,savedata=False,run=VCA.VCADTBT))
    vca.register(VCA.CPFF(name='FF',task='FF',parameters={'afm':0.20},cf=U/2,BZ=KSpace(reciprocals=lattice.reciprocals,nk=100),run=VCA.VCACPFF))
    #vca.register(VCA.CPFF(name='CP',task='CP',cf=0.5,BZ=KSpace(reciprocals=lattice.reciprocals,nk=100),options={'x0':1.0,'x_tol':10**-6,'maxiter':20},run=VCA.VCACPFF))
    vca.summary()
    print
