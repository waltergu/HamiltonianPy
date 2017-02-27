'''
Tree test.
'''

__all__=['test_tree']

from HamiltonianPy.Misc.Tree import *

def test_tree():
    print 'test_tree'
    A=Tree()
    A.add_leaf(None,'L0-0',1.0)
    print 'A: %s'%A

    B=Tree()
    B.add_leaf(None,'L1-0',2.0)
    B.add_leaf('L1-0','L2-0',3.0)
    B.add_leaf('L1-0','L2-1',4.0)
    print 'B: %s'%B

    C=Tree('L1-1',5.0)
    C.add_leaf('L1-1','L2-2',6.0)
    C.add_leaf('L1-1','L2-3',7.0)
    print 'C: %s'%C
    
    A.add_subtree(B,parent='L0-0')
    A.add_subtree(C,parent='L0-0')
    print 'A: %s'%A

    for s in A.expand(mode=Tree.WIDTH,return_form=Tree.PAIR):
        print s

    print "A.ancestor('L2-0',generation=2): %s"%A.ancestor('L2-0',generation=2)
    print "A.parent('L1-0'): %s"%A.parent('L1-0')
    print "A.children('L1-1'): %s"%A.children('L1-1')
    print "A.siblings('L1-1'): %s"%A.siblings('L1-1')
    print "A.descendants('L0-0',generation=2): %s"%A.descendants('L0-0',generation=2)

    A.remove_subtree('L1-0')
    for s in A.expand(mode=Tree.WIDTH,return_form=Tree.PAIR):
        print s
    print

    D=A.subtree(node='L1-1')
    for s in D.expand():
        print s
    print
