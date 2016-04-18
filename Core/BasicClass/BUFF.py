#def SuperLattice(name,sublattices,vectors=[],nneighbour=1,priority='PNSCO'):
#    '''
#    This function returns the union of sublattices.
#    Parameters:
#        name: string
#            The name of the super-lattice.
#        sublattices: list of Lattice
#            The sub-lattices of the super-lattice.
#        vectors: list of 1D ndarray, optional
#            The translation vectors of the super-lattice.
#        nneighbour: integer,optional
#            The highest order of neighbours.
#        priority: string, optional
#            The sequence priority of the allowed indices that can be defined on this super-lattice.
#    Returns:
#        result: Lattice
#            The super-lattice.
#    '''
#    result=object.__new__(Lattice)
#    result.name=name
#    result.points={}
#    for lattice in sublattices:
#        result.points.update(lattice.points)
#    result.vectors=vectors
#    result.reciprocals=reciprocals(vectors)
#    result.nneighbour=nneighbour
#    result.bonds=[b for bs in bonds(result.points,vectors,nneighbour) for b in bs]
#    result.priority=priority
#    result.sublattices=sublattices
#    return result




    #def table(self,nambu=False,priority=None):
    #    '''
    #    This method returns a Table instance that contains all the allowed indices which can be defined on this point.
    #    '''
    #    return self.struct.table(site=self.site,scope=self.scope,nambu=nambu,priority=priority)


#def translation(points,vector):
#    '''
#    This function returns the translated points.
#    Parameters:
#        points: list of Point
#            The original points.
#        vector: 1D ndarray
#            The translation vector.
#    Returns:
#        result: list of Point
#            The translated points.
#    '''
#    result=[]
#    for p in points:
#        result.append(Point(id=deepcopy(p.id),rcoord=p.rcoord+vector,icoord=deepcopy(p.icoord)))
#    return result

#def rotation(points=None,coords=None,angle=0,axis=None,center=None):
#    '''
#    This function returns the rotated points or coords.
#    Parameters:
#        points: list of Point
#            The original points.
#        coords: list of 1D ndarray
#            The original coords.
#        angle: float
#            The rotated angle
#        axis: 1D array-like, optional
#            The rotation axis. Default the z-axis.
#            Not supported yet.
#        center: 1D array-like, optional
#            The center of the axis. Defualt the origin.
#    Returns:
#        result: list of Point/1D ndarray
#            The rotated points or coords.
#    Note: points and coords cannot be both None or not None.
#    '''
#    if points is None and coords is None:
#        raise ValueError('rotation error: both points and coords are None.')
#    if points is not None and coords is not None:
#        raise ValueError('rotation error: both points and coords are not None.')
#    result=[]
#    if center is None: center=0
#    m11=cos(angle);m21=-sin(angle);m12=-m21;m22=m11
#    m=array([[m11,m12],[m21,m22]])
#    if points is not None:
#        for p in points:
#            result.append(Point(id=deepcopy(p.id),rcoord=dot(m,p.rcoord-center)+center,icoord=deepcopy(p.icoord)))
#    if coords is not None:
#        for coord in coords:
#            result.append(dot(m,coord-center)+center)
#    return result

        #priority: string
        #    The sequence priority of the allowed indices that can be defined on this lattice.
        #    Default value 'PNSCO', where 'P','N','S','C','O' stands for 'scope', 'nambu', 'spin', 'site' and 'orbital' respectively.

    #def table(self,nambu=False):
    #    '''
    #    Return a Table instance that contains all the allowed indices which can be defined on this lattice.
    #    '''
    #    return union([p.table(nambu=nambu) for p in self.points.itervalues()],key=lambda value: value.to_tuple(indication=self.priority))
