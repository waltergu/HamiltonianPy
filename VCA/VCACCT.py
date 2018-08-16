'''
=====================================
CPT and VCA with concatenated lattice
=====================================

CPT and VCA with concatenated lattice, including:
    * classes: SubVCA, VCACCT
    * functions: VCACCTGFP, VCACCTGF
'''

__all__=['SubVCA','VCACCT','VCACCTGFP','VCACCTGF']

from .VCA import *
from copy import deepcopy
from collections import Counter,OrderedDict
import numpy as np
import HamiltonianPy as HP
import HamiltonianPy.ED as ED
import itertools as it

class SubVCA(ED.FED):
    '''
    A subsystem of concatenated VCA.

    Attributes
    ----------
    sector,sectors,lattice,config,terms,dtype,operators :
        Inherited from ED.FED. See ED.FED for details.
    weiss : list of Term
        The Weiss terms of the system.
    baths : list of Term
        The bath terms of the system.
    hgenerator,wgenerator,bgenerator : Generator
        The generator of the original/Weiss/bath part of the subsystem.
    '''

    def __init__(self,sectors,lattice,config,terms=(),weiss=(),baths=(),dtype=np.complex128,log=None,**karg):
        '''
        Constructor.

        Parameters
        ----------
        sectors,lattice,config,terms,weiss,baths,dtype:
            See ED.FED.__init__ for details.
        log : Log, optional
            The log of the subsystem.
        '''
        self.sectors=set(sectors)
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.weiss=weiss
        self.baths=baths
        self.dtype=dtype
        self.sector=None
        self.hgenerator=HP.Generator(
            bonds=      [bond for bond in lattice.bonds if bond.isintracell() and 'BATH' not in bond.spoint.pid.scope and 'BATH' not in bond.epoint.pid.scope],
            config=     config,
            table=      config.table(mask=('nambu',)),
            terms=      deepcopy(terms),
            dtype=      dtype,
            half=       True
            )
        self.wgenerator=HP.Generator(
            bonds=      [bond for bond in lattice.bonds if 'BATH' not in bond.spoint.pid.scope and 'BATH' not in bond.epoint.pid.scope],
            config=     config,
            table=      config.table(mask=('nambu',)),
            terms=      deepcopy(weiss),
            dtype=      dtype,
            half=       True
            )
        self.bgenerator=HP.Generator(
            bonds=      [bond for bond in lattice.bonds if 'BATH' in bond.spoint.pid.scope or 'BATH' in bond.epoint.pid.scope],
            config=     config,
            table=      config.table(mask=('nambu',)),
            terms=      deepcopy(baths),
            dtype=      dtype,
            half=       True
            )
        if self.map is None: self.parameters.update(OrderedDict((term.id,term.value) for term in it.chain(terms,weiss,baths)))
        self.operators=self.hgenerator.operators+self.wgenerator.operators+self.bgenerator.operators
        self.cache={}
        if log is None:
            self.logging()
        else:
            self.log=log

    def matrix(self,sector,reset=True):
        '''
        The matrix representation of the Hamiltonian.

        Parameters
        ----------
        sector : str
            The sector of the matrix representation of the Hamiltonian.
        reset : logical, optional
            True for resetting the matrix cache and False for not.

        Returns
        -------
        csr_matrix
            The matrix representation of the Hamiltonian.
        '''
        if reset:
            self.hgenerator.setmatrix(sector,HP.foptrep,sector,transpose=False,dtype=self.dtype)
            self.wgenerator.setmatrix(sector,HP.foptrep,sector,transpose=False,dtype=self.dtype)
            self.bgenerator.setmatrix(sector,HP.foptrep,sector,transpose=False,dtype=self.dtype)
        self.sector=sector
        matrix=self.hgenerator.matrix(sector)+self.wgenerator.matrix(sector)+self.bgenerator.matrix(sector)
        return matrix.T+matrix.conjugate()

    def removesector(self,sector,newcurrent=None):
        '''
        Remove a sector from the engine.

        Parameters
        ----------
        sector : hashable
            The sector to be removed.
        newcurrent : hashable, optional
            The new sector of the engine.
        '''
        self.sectors.remove(sector)
        if newcurrent is not None: self.sector=newcurrent
        self.hgenerator.removematrix(sector)
        self.wgenerator.removematrix(sector)
        self.bgenerator.removematrix(sector)

    def update(self,**karg):
        '''
        Update the engine.
        '''
        if len(karg)>0:
            super(ED.ED,self).update(**karg)
            data=self.data
            self.hgenerator.update(**data)
            self.wgenerator.update(**data)
            self.bgenerator.update(**data)
            self.operators=self.hgenerator.operators+self.wgenerator.operators+self.bgenerator.operators

class VCACCT(VCA):
    '''
    This class implements the algorithm of the variational cluster approach of an electron system composed of several subsystems.

    Attributes
    ----------
    cell,lattice,config,terms,weiss,baths,mask,dtype,pthgenerator,ptwgenerator,ptbgenerator,pthoperators,ptwoperators,ptboperators,periodization,cache :
        Inherited from VCA. See VCA for details.
    groups : list of hashable object
        The groups of the components of the system.
    subsystems : dict in the form (key,value)
        * key: any hashable object
            The group of the subsystems.
        * value: SubVCA
            A representative subsystem of the same group.
    '''

    def __init__(self,cgf,cell,lattice,config,terms=(),weiss=(),baths=(),mask=('nambu',),subsystems=None,dtype=np.complex128,**karg):
        '''
        Constructor.

        Parameters
        ----------
        cgf,cell,lattice,config,term,weiss,baths,mask,dtype :
            See VCA.__init__ for details.
        subsystems : list of dict, with each dict
            * entry 'sectors': iterable of FBasis
                The occupation number bases of the subsystem.
            * entry 'lattice': Lattice
                The lattice of the subsystem.
            * entry 'group': any hashable object, optional
                The group of the subsystem.
        '''
        assert isinstance(cgf,VGF)
        subconfigs=[HP.IDFConfig(priority=config.priority,pids=subsystem['lattice'].pids,map=config.map) for subsystem in subsystems]
        cgf.resetopts(HP.fspoperators(HP.Table.union([subconfig.table() for subconfig in subconfigs]),lattice))
        self.preload(cgf)
        self.preload(HP.GF(operators=HP.fspoperators(HP.IDFConfig(priority=config.priority,pids=cell.pids,map=config.map).table(),cell),dtype=cgf.dtype))
        self.cell=cell
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.weiss=weiss
        self.baths=baths
        self.mask=mask
        self.dtype=dtype
        if self.map is None: self.parameters.update(OrderedDict((term.id,term.value) for term in it.chain(terms,weiss,baths)))
        self.logging()
        self.groups=[subsystem.get('group',subsystem['lattice'].name) for subsystem in subsystems]
        self.subsystems={}
        extras={key:value for key,value in karg.items() if key!='name'}
        attrs={attr:vars(cgf)[attr] for attr in set(vars(cgf))-{'name','parameters','virgin','operators','k','omega','prepare','run'}}
        for group in set(self.groups):
            index=self.groups.index(group)
            subsystem,subconfig=subsystems[index],subconfigs[index]
            subsectors,sublattice=subsystem['sectors'],subsystem['lattice']
            self.subsystems[group]=SubVCA(
                    name=           group,
                    cgf=            VGF(),
                    sectors=        subsectors,
                    lattice=        sublattice,
                    config=         subconfig,
                    terms=          terms,
                    weiss=          weiss,
                    baths=          baths,
                    dtype=          dtype,
                    log=            self.log,
                    **extras
                    )
            self.subsystems[group].add(VGF(
                    name=           'gf',
                    operators=      HP.fspoperators(subconfig.table(),sublattice),
                    prepare=        ED.EDGFP,
                    run=            ED.EDGF,
                    **attrs
                    ))
        self.pthgenerator=HP.Generator(
            bonds=      [bond for bond in lattice.bonds if 'BATH' not in bond.spoint.pid.scope and 'BATH' not in bond.epoint.pid.scope
                                                        and (not bond.isintracell() or bond.spoint.pid.scope!=bond.epoint.pid.scope)],
            config=     config,
            table=      HP.Table.union([subconfig.table(mask=mask) for subconfig in subconfigs]),
            terms=      [deepcopy(term) for term in terms if isinstance(term,HP.Quadratic)],
            dtype=      dtype,
            half=       True
            )
        self.ptwgenerator=HP.Generator(
            bonds=      [bond for bond in lattice.bonds if 'BATH' not in bond.spoint.pid.scope and 'BATH' not in bond.epoint.pid.scope
                                                        and bond.spoint.pid.scope==bond.epoint.pid.scope],
            config=     config,
            table=      HP.Table.union([subconfig.table(mask=mask) for subconfig in subconfigs]),
            terms=      deepcopy(weiss),
            dtype=      dtype,
            half=       True
            )
        self.ptbgenerator=HP.Generator(
            bonds=      [bond for bond in lattice.bonds if 'BATH' in bond.spoint.pid.scope or 'BATH' in bond.epoint.pid.scope],
            config=     config,
            table=      HP.Table.union([subconfig.table(mask=mask) for subconfig in subconfigs]),
            terms=      deepcopy(baths),
            dtype=      dtype,
            half=       True
            )
        self.pthoperators=self.pthgenerator.operators
        self.ptwoperators=self.ptwgenerator.operators
        self.ptboperators=self.ptbgenerator.operators
        self.periodize()
        self.cache={}

    def update(self,**karg):
        '''
        Update the engine.
        '''
        if len(karg)>0:
            self.CGF.virgin=True
            for subsystem in self.subsystems.values():
                subsystem.update(**karg)
            super(ED.ED,self).update(**karg)
            data=self.data
            self.pthgenerator.update(**data)
            self.ptwgenerator.update(**data)
            self.ptbgenerator.update(**data)
            self.pthoperators=self.pthgenerator.operators
            self.ptwoperators=self.ptwgenerator.operators
            self.ptboperators=self.ptbgenerator.operators

def VCACCTGFP(engine,app):
    '''
    This method prepares the cluster Green's function.
    '''
    app.gse=0.0
    counter=Counter(engine.groups)
    for group,subsystem in engine.subsystems.items():
        subsystem.apps['gf'].prepare(subsystem,subsystem.apps['gf'])
        app.gse+=subsystem.apps['gf'].gse*counter[group]

def VCACCTGF(engine,app):
    '''
    This method calculate the cluster Green's function.
    '''
    cgf=engine.records[app.name]
    if app.omega is not None:
        gfs={}
        for group,subsystem in engine.subsystems.items():
            gf=subsystem.apps['gf']
            gf.omega=app.omega
            gfs[group]=gf.run(subsystem,gf)
        if cgf is None:
            cgf=np.zeros((app.nopt,app.nopt),dtype=app.dtype)
        else:
            cgf[...]=0
        row,col=0,0
        for gf in (gfs[group] for group in engine.groups):
            cgf[row:row+gf.shape[0],col:col+gf.shape[1]]=gf
            row+=gf.shape[0]
            col+=gf.shape[1]
        engine.records[app.name]=cgf
    return cgf
