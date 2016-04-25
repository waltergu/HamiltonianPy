from VCAPy import *
from scipy.linalg import block_diag
class VCACCT(VCA):
    '''
    '''
    def __init__(self,ensemble='c',filling=0.5,mu=0,nspin=1,cell=None,lattice=None,subsystems=None,terms=None,weiss=None,nambu=False,**karg):
        self.ensemble=ensemble
        self.filling=filling
        self.mu=mu
        if self.ensemble.lower()=='c':
            self.name.update(const={'filling':self.filling})
        elif self.ensemble.lower()=='g':
            self.name.update(alter={'mu':self.mu})
        self.cell=cell
        self.lattice=lattice
        self.terms=terms
        self.weiss=weiss
        self.nambu=nambu
        self.groups={}
        self.subsystems={}
        for i,subsystem in enumerate(subsystems):
            sub_filling=filling if 'filling' not in subsystem else subsystem['filling']
            sub_mu=mu if 'mu' not in subsystem else subsytem['mu']
            sub_basis=subsystem['basis']
            sub_lattice=subsystem['lattice']
            group=sub_lattice.name if 'group' not in subsystem else subsystem['group']
            if group not in self.groups:
                self.groups[group]=[]
            self.groups[group].append(sub_lattice.name)
            self.subsystems[sub_lattice.name]=ONR(
                    name=       sub_lattice.name,
                    ensemble=   ensemble,
                    filling=    sub_filling,
                    mu=         sub_mu,
                    basis=      sub_basis,
                    nspin=      nspin,
                    lattice=    sub_lattice,
                    terms=      terms if weiss is None else terms+weiss,
                    nambu=      nambu,
                    **{key:karg[key] for key in karg if key!='name'}
                )
            if i==0: flag=self.subsystems[sub_lattice.name].nspin
            if flag!=self.subsystems[sub_lattice.name].nspin:
                raise ValueError("VCACCT init error: all the subsystems must have the same nspin.")
        self.nspin=flag
        self.generators={}
        self.generators['pt_h']=Generator(
                    bonds=      [bond for bond in lattice.bonds if not bond.is_intra_cell() or bond.spoint.scope!=bond.epoint.scope],
                    table=      lattice.table(nambu=nambu) if self.nspin==2 else subset(lattice.table(nambu=nambu),mask=lambda index: True if index.spin==0 else False),
                    terms=      [term for term in terms if isinstance(term,Quadratic)],
                    nambu=      nambu,
                    half=       True
                    )
        self.generators['pt_w']=Generator(
                    bonds=      [bond for bond in lattice.bonds if bond.is_intra_cell() and bond.spoint.scope==bond.epoint.scope],
                    table=      lattice.table(nambu=nambu) if self.nspin==2 else subset(lattice.table(nambu=nambu),mask=lambda index: True if index.spin==0 else False),
                    terms=      None if weiss is None else [term*(-1) for term in weiss],
                    nambu=      nambu,
                    half=       True
                    )
        self.name.update(const=self.subsystems[self.subsystems.keys()[0]].generators['h'].parameters['const'])
        self.name.update(alter=self.subsystems[self.subsystems.keys()[0]].generators['h'].parameters['alter'])
        self.operators={}
        self.set_operators()
        self.clmap={}
        self.set_clmap()
        self.cache={}

    def set_operators(self):
        self.set_operators_perturbation()
        self.set_operators_single_particle()
        self.set_operators_cell_single_particle()

    def update(self,**karg):
        '''
        Update the alterable operators, such as the weiss terms.
        '''
        for subsystem in self.subsystems.itervalues():
            subsystem.update(**karg)
        for generator in self.generators.itervalues():
            generator.update(**karg)
        self.name.update(alter=self.generators['pt_h'].parameters['alter'])
        self.set_operators_perturbation()

    def gf(self,omega=None):
        buff=[]
        for group in self.groups.itervalues():
            buff.extend([self.subsystems[group[0]].gf(omega)]*len(group))
        return block_diag(*buff)

def VCACCTGFC(engine,app):
    buff=deepcopy(app)
    buff.run=ONRGFC
    for group in engine.groups.itervalues():
        for i,name in enumerate(group):
            if i==0:
                engine.subsystems[name].addapps('GFC',buff)
                engine.subsystems[name].runapps('GFC')
