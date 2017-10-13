'''
=====================================
CPT and VCA with concatenated lattice
=====================================

CPT and VCA with concatenated lattice, including:
    * classes: VCACCT
    * functions: VCACCTGFP, VCACCTGF
'''

__all__=['VCACCT','VCACCTGFP','VCACCTGF']

from VCA import *
from copy import deepcopy
from collections import Counter,OrderedDict
import numpy as np
import HamiltonianPy as HP
import HamiltonianPy.ED as ED

class VCACCT(VCA):
    '''
    This class implements the algorithm of the variational cluster approach of an electron system composed of several subsystems.

    Attributes
    ----------
    preloads,cell,lattice,config,terms,weiss,mask,dtype,pthgenerator,ptwgenerator,pthoperators,ptwoperators,periodization,matrix,cache :
        Inherited from VCA. See VCA for details.
    groups : list of hashable object
        The groups of the components of the system.
    subsystems : dict in the form (key,value)
        * key: any hashable object
            The group of the subsystems.
        * value: ED.ED
            A representative subsystem of the same group.
    '''

    def __init__(self,cgf,cell,lattice,config,terms=(),weiss=(),mask=('nambu',),subsystems=None,dtype=np.complex128,**karg):
        '''
        Constructor.

        Parameters
        ----------
        cgf,cell,lattice,config,terms,weiss,mask :
            See VCA.__init__ for details.
        subsystem: dict
            * entry 'basis': FBasis
                The occupation number basis of the subsystem.
            * entry 'lattice': Lattice
                The lattice of the subsystem.
            * entry 'group': any hashable object, optional
                The group of the subsystem.
        '''
        cellconfig=HP.IDFConfig(priority=config.priority,pids=cell.pids,map=config.map)
        self.preloads.extend([cgf,HP.GF(operators=HP.fspoperators(cellconfig.table(),cell),dtype=cgf.dtype)])
        self.cell=cell
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.weiss=weiss
        self.mask=mask
        self.dtype=dtype
        self.groups=[subsystem.get('group',subsystem['lattice'].name) for subsystem in subsystems]
        self.subsystems={}
        extras={key:value for key,value in karg.iteritems() if key!='name'}
        for group in set(self.groups):
            subsystem=subsystems[self.groups.index(group)]
            subbasis,sublattice=subsystem['basis'],subsystem['lattice']
            subconfig=HP.IDFConfig(priority=config.priority,pids=subsystem['lattice'].pids,map=config.map)
            self.subsystems[group]=ED.FED(
                    name=           group,
                    basis=          subbasis,
                    lattice=        sublattice,
                    config=         subconfig,
                    terms=          deepcopy(terms+weiss),
                    dtype=          dtype,
                    **extras
                    )
            attributes={attr:vars(cgf)[attr] for attr in set(vars(cgf))-{'name','parameters','virgin','operators','gf','k','omega','prepare','run'}}
            gf=ED.GF(
                    name=           'gf',
                    operators=      HP.fspoperators(subconfig.table(),lattice),
                    prepare=        ED.EDGFP,
                    run=            ED.EDGF,
                    **attributes
                    )
            self.subsystems[group].register(gf,run=False)
        if self.map is None: self.parameters.update(OrderedDict((term.id,term.value) for term in terms+weiss))
        self.pthgenerator=HP.Generator(
                    bonds=          [bond for bond in lattice.bonds if not bond.isintracell() or bond.spoint.pid.scope!=bond.epoint.pid.scope],
                    config=         config,
                    table=          config.table(mask=mask),
                    terms=          [term for term in terms if isinstance(term,HP.Quadratic)],
                    dtype=          dtype,
                    half=           True
                    )
        self.ptwgenerator=HP.Generator(
                    bonds=          [bond for bond in lattice.bonds if bond.isintracell() and bond.spoint.pid.scope==bond.epoint.pid.scope],
                    config=         config,
                    table=          config.table(mask=mask),
                    terms=          [term*(-1) for term in weiss],
                    dtype=          dtype,
                    half=           True
                    )
        self.pthoperators=self.pthgenerator.operators
        self.ptwoperators=self.ptwgenerator.operators
        self.periodize()
        self.cache={}

    def update(self,**karg):
        '''
        Update the engine.
        '''
        if len(karg)>0:
            for subsystem in self.subsystems.itervalues():
                subsystem.update(**karg)
            super(ED.ED,self).update(**karg)
            karg=self.data(karg)
            self.pthgenerator.update(**karg)
            self.ptwgenerator.update(**karg)
            self.pthoperators=self.pthgenerator.operators
            self.ptwoperators=self.ptwgenerator.operators

def VCACCTGFP(engine,app):
    '''
    This method prepares the cluster Green's function.
    '''
    app.gse=0.0
    counter=Counter(engine.groups)
    for group,subsystem in engine.subsystems.iteritems():
        subsystem.apps['gf'].prepare(subsystem,subsystem.apps['gf'])
        app.gse+=subsystem.apps['gf'].gse*counter[group]

def VCACCTGF(engine,app):
    '''
    This method calculate the cluster Green's function.
    '''
    if app.omega is not None:
        gfs={}
        for group,subsystem in engine.subsystems.iteritems():
            gf=subsystem.apps['gf']
            gf.omega=app.omega
            gfs[group]=gf.run(subsystem,gf)
        app.gf[...],row,col=0,0,0
        for group in engine.groups:
            app.gf[row:row+gfs[group].shape[0],col:col+gfs[group].shape[1]]=gfs[group]
            row+=gfs[group].shape[0]
            col+=gfs[group].shape[1]
    return app.gf
