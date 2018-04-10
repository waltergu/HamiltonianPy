'''
Tree test (9 tests in total).
'''

__all__=['tree']

from HamiltonianPy.Misc.Tree import *
from unittest import TestCase,TestLoader,TestSuite

class TestTree(TestCase):
    def setUp(self):
        self.tree=Tree('L0-0',1.0)
        self.tree.add_leaf('L0-0','L1-0',2.0)
        self.tree.add_leaf('L1-0','L2-0',3.0)
        self.tree.add_leaf('L1-0','L2-1',4.0)
        self.tree.add_leaf('L0-0','L1-1',5.0)
        self.tree.add_leaf('L1-1','L2-2',6.0)
        self.tree.add_leaf('L1-1','L2-3',7.0)

    def test_expand(self):
        depth=[('L0-0',1.0),('L1-0',2.0),('L2-0',3.0),('L2-1',4.0),('L1-1',5.0),('L2-2',6.0),('L2-3',7.0)]
        width=[('L0-0',1.0),('L1-0',2.0),('L1-1',5.0),('L2-0',3.0),('L2-1',4.0),('L2-2',6.0),('L2-3',7.0)]
        self.assertEqual(list(self.tree.expand(mode=Tree.DEPTH,return_form=Tree.PAIR)),depth)
        self.assertEqual(list(self.tree.expand(mode=Tree.WIDTH,return_form=Tree.PAIR)),width)

    def test_parent(self):
        self.assertEqual(self.tree.parent('L1-0'),'L0-0')

    def test_children(self):
        self.assertEqual(self.tree.children('L1-1'),['L2-2','L2-3'])

    def test_siblings(self):
        self.assertEqual(self.tree.siblings('L1-1'),['L1-0'])

    def test_ancestor(self):
        self.assertEqual(self.tree.ancestor('L2-0',generation=2),'L0-0')
        self.assertEqual(self.tree.ancestor('L2-2',generation=1),'L1-1')

    def test_descendants(self):
        self.assertEqual(self.tree.descendants('L0-0',generation=2),['L2-0','L2-1','L2-2','L2-3'])
        self.assertEqual(self.tree.descendants('L0-0',generation=1),['L1-0','L1-1'])

    def test_subtree(self):
        pairs=[('L1-0',2.0),('L2-0',3.0),('L2-1',4.0)]
        self.assertEqual(list(self.tree.subtree(node='L1-0').expand(mode=Tree.WIDTH,return_form=Tree.PAIR)),pairs)

    def test_add_subtree(self):
        subtree=Tree('L1-2',8.0)
        subtree.add_leaf('L1-2','L2-4',9.0)
        subtree.add_leaf('L1-2','L2-5',10.0)
        self.tree.add_subtree(subtree,parent='L0-0')
        pairs=[('L0-0',1.0),('L1-0',2.0),('L2-0',3.0),('L2-1',4.0),('L1-1',5.0),('L2-2',6.0),('L2-3',7.0),('L1-2',8.0),('L2-4',9.0),('L2-5',10.0)]
        self.assertEqual(list(self.tree.expand(mode=Tree.DEPTH,return_form=Tree.PAIR)),pairs)

    def test_remove_subtree(self):
        pairs=[('L0-0',1.0),('L1-1',5.0),('L2-2',6.0),('L2-3',7.0)]
        self.tree.remove_subtree('L1-0')
        self.assertEqual(list(self.tree.expand(mode=Tree.WIDTH,return_form=Tree.PAIR)),pairs)

tree=TestSuite([
            TestLoader().loadTestsFromTestCase(TestTree),
            ])
