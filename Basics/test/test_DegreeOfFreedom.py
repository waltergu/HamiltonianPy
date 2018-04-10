'''
Degree of freedom test (3 tests in total).
'''

__all__=['degreeoffreedom']

from HamiltonianPy.Basics.DegreeOfFreedom import Table
from unittest import TestCase,TestLoader,TestSuite

class TestTable(TestCase):
    def setUp(self):
        self.table=Table(['i1','i2'])

    def test_union(self):
        another=Table(['i4','i3'])
        union=Table.union([self.table,another],key=lambda key: key[1])
        result=Table(['i1','i2','i3','i4'])
        self.assertEqual(union,result)

    def test_reverse_table(self):
        reversed_table=self.table.reversed_table
        result=Table()
        result[0]='i1'
        result[1]='i2'
        self.assertEqual(reversed_table,result)

    def test_subset(self):
        subset=self.table.subset(select=lambda key: True if key!='i1' else False)
        result=Table(['i2'])
        self.assertEqual(subset,result)

degreeoffreedom=TestSuite([
                        TestLoader().loadTestsFromTestCase(TestTable),
                        ])
