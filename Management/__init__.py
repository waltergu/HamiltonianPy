'''
============
Introduction
============

This subpackage defines several command line commands to manage the construction and maintenance of a project.
Supported commands include:
    * init
        Initialize a project
    * git
        Greate a git repository for the current project
    * add
        Add templates to the current project
'''

import os
from Commands import *

class Manager(object):
    '''

    '''

    def __init__(self,args):
        self.args=args
        self.prog_name=os.path.basename(self.args[0])
        print self.prog_name

    def execute(self):
        try:
            subcommand=self.args[1]
        except IndexError:
            subcommand='help'

