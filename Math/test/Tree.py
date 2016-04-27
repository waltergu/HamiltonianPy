'''
Tree test.
'''

__all__=['test_tree']

from HamiltonianPP.Math.TreePy import *

def test_tree():
    A=Tree()
    A['L0-0']=Node(1.0)
    print 'A: %s'%A

    B=Tree()
    B['L1-0']=Node(2.0)
    B['L2-0']=Node(3.0,parent='L1-0')
    B['L2-1']=Node(4.0,parent='L1-0')
    print 'B: %s'%B

    C=Tree('L1-1',Node(5.0))
    C['L2-2']=Node(6.0,parent='L1-1')
    C['L2-3']=Node(7.0,parent='L1-1')
    print 'C: %s'%C
    
    A.add_subtree(B,parent='L0-0')
    A.add_subtree(C,parent='L0-0')
    print 'A: %s'%A

    for s in A.expand(mode=Tree.WIDTH,return_form=Tree.PAIR):
        print s

    print "A.parent('L1-0'): %s"%A.parent('L1-0')
    print "A.children('L1-1'): %s"%A.children('L1-1')

    A.remove_subtree('L1-0')
    for s in A.expand(mode=Tree.WIDTH,return_form=Tree.PAIR):
        print s

    D=A.subtree(nid='L1-1')
    for s in D.expand():
        print s

    D['L1-1'].update(111.0)
    print "A['L1-1']: %s"%A['L1-1']
