    #def table(self,nambu=False,priority=None):
    #    '''
    #    This method returns a Table instance that contains all the allowed indices which can be defined on this point.
    #    '''
    #    return self.struct.table(site=self.site,scope=self.scope,nambu=nambu,priority=priority)

    #priority: string
    #    The sequence priority of the allowed indices that can be defined on this lattice.
    #    Default value 'PNSCO', where 'P','N','S','C','O' stands for 'scope', 'nambu', 'spin', 'site' and 'orbital' respectively.

    #def table(self,nambu=False):
    #    '''
    #    Return a Table instance that contains all the allowed indices which can be defined on this lattice.
    #    '''
    #    return union([p.table(nambu=nambu) for p in self.points.itervalues()],key=lambda value: value.to_tuple(indication=self.priority))
