'''
Degree of freedom test.
'''

__all__=['test_deg_fre']

from HamiltonianPy.Basics.DegreeOfFreedom import *

def test_deg_fre():
    test_table()

def test_table():
    print 'test_table'
    a=Table(['i1','i2'])
    b=Table(['i3','i4'])
    c=Table.union([a,b],key=lambda key: key[1])
    print 'a: %s'%a
    print 'b: %s'%b
    print 'c=union(a,b): %s'%c
    print 'c.reverse_table: %s'%c.reversed_table
    print 'c["i4"]: %s'%c['i4']
    print 'c.subset: %s'%c.subset(select=lambda key: True if key!='i1' else False)
    print
