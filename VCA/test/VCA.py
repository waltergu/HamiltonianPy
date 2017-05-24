'''
VCA test.
'''

__all__=['test_vca']

from HamiltonianPy import *
import HamiltonianPy.ED as ED
import HamiltonianPy.VCA as VCA
import numpy as np
import itertools as it

def test_vca():
    print 'test_vca'
    t,U=-1.0,8.0
    m=2;n=2;nspin=2
    name='%s%s%s'%('WG',m,n)
    a1,a2=np.array([1.0,0.0]),np.array([0.0,1.0])
    lattice=Lattice(name=name,rcoords=tiling([np.array([0.0,0.0])],vectors=[a1,a2],translations=it.product(xrange(m),xrange(n))),vectors=[a1*m,a2*n])
    cell=Lattice(name=name,rcoords=[np.array([0.0,0.0])],vectors=[a1,a2])
    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=lambda pid: Fermi(atom=0,norbital=1,nspin=2,nnambu=1))
    cgf=ED.GF(operators=fspoperators(config.table().subset(ED.GF.select(nspin)),lattice),nspin=nspin,nstep=200,save_data=False,prepare=ED.EDGFP,run=ED.EDGF)
    afm=lambda bond: 1 if bond.spoint.pid.site in (0,3) else -1
    vca=VCA.VCA(
            name=       name,
            cgf=        cgf,
            basis=      FBasis(up=(m*n,m*n/2),down=(m*n,m*n/2)),
            cell=       cell,
            lattice=    lattice,
            config=     config,
            terms=      [Hopping('t',t,neighbour=1),Hubbard('U',U)],
            weiss=      [Onsite('afm',0.0,indexpacks=sigmaz('sp'),amplitude=afm,modulate=lambda **karg:karg.get('afm',None))]
            )
    gp=GP(name='GP',mu=U/2,BZ=square_bz(reciprocals=vca.lattice.reciprocals,nk=100),run=VCA.VCAGP)
    vca.register(VCA.GPM(name='afm-1',fout='afm.dat',BS={'afm':0.1},method='BFGS',options={'disp':True},save_data=False,dependences=[gp],run=VCA.VCAGPM))
    vca.register(VCA.GPM(name='afm-2',BS=BaseSpace(('afm',np.linspace(0.0,0.3,16))),save_data=False,plot=True,dependences=[gp],run=VCA.VCAGPM))
    vca.register(VCA.EB(name='EB',parameters={'afm':0.20},path=square_gxm(nk=100),mu=U/2,emax=6.0,emin=-6.0,eta=0.05,ne=400,save_data=False,run=VCA.VCAEB))
    vca.register(DOS(name='DOS',parameters={'afm':0.20},BZ=square_bz(reciprocals=vca.lattice.reciprocals,nk=20),mu=U/2,emin=-10,emax=10,ne=400,eta=0.05,save_data=False,plot=True,show=True,run=VCA.VCADOS))
    vca.register(FS(name='FS',parameters={'afm':0.20},mu=U/2,BZ=square_bz(nk=100),save_data=False,run=VCA.VCAFS))
    vca.register(VCA.OP(name='OP',parameters={'afm':0.20},mu=U/2,terms=vca.weiss,BZ=square_bz(reciprocals=vca.lattice.reciprocals,nk=100),run=VCA.VCAOP))
    vca.register(VCA.DTBT(name='DTBT',parameters={'afm':0.00},path=square_gxm(nk=100),mu=U/2,save_data=False,plot=True,run=VCA.VCADTBT))
    vca.register(VCA.CPFF(name='FF',task='FF',parameters={'afm':0.20},mu=U/2,BZ=square_bz(reciprocals=vca.lattice.reciprocals,nk=100),p=0.5,run=VCA.VCACPFF))
    #vca.register(VCA.CPFF(name='CP',task='CP',filling=0.5,mu=1.0,BZ=square_bz(reciprocals=vca.lattice.reciprocals,nk=100),error=10**-6,run=VCA.VCACPFF))
    vca.summary()
    print
