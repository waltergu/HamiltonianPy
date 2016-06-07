'''
Degree of freedom test.
'''

__all__=['test_deg_fre']

from HamiltonianPy.Basics.Geometry import *
from HamiltonianPy.Basics.DegreeOfFreedom import *

def test_deg_fre():
    test_table()

def test_table():
    print 'test_table'
    a=Table({'i1':0,'i2':1})
    b=Table({'i3':0,'i4':1})
    c=union([a,b],key=lambda key: key[1])
    print 'a: %s'%a
    print 'b: %s'%b
    print 'union(a,b)(c): %s'%c
    print 'reverse_table(c): %s'%reversed_table(c)
    print 'c["i4"]: %s'%c['i4']
    print 'subset: %s'%subset(c,mask=lambda key: True if key!='i1' else False)
    print
