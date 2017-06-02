'''
This subpackage implements the construction and maintenance of a project through command line commands, including:
    * classes: Manager
    * functions: init, git, add
'''

import os
import Template
from argparse import ArgumentParser

__all__=['Manager','init','git','add']

class Manager(object):
    '''
    The manager of project construction and maitenance.

    Attributes
    ----------
    args : list of str
        The arguments passed to this manager.
    parser : ArgumentParser
        The argument parser.
    '''

    def __init__(self,args):
        '''
        Constructor.

        Parameters
        ----------
        args : list of str
            The args passed to this manager.
        '''
        self.args=args
        self.parser=ArgumentParser(prog=os.path.basename(args[0]))
        self.init()
        self.git()
        self.add()

    def add_subcommand(self,name,subcommand):
        '''
        Add a subcommand to the manager.

        Parameters
        ----------
        name : str
            The name of the subcommand.
        subcommand : callable
            The function that implements the subcommand.

        Returns
        -------
        ArgumentParser
            The subparser corresponding to the subcommand.
        '''
        actions=self.parser.add_subparsers() if self.parser._subparsers is None else self.parser._get_positional_actions()[0]
        subparser=actions.add_parser(name)
        subparser.set_defaults(subcommand=subcommand)
        return subparser

    def init(self):
        '''
        Subcommand init, which initializes the project.
        '''
        subcommand=self.add_subcommand('init',init)
        subcommand.add_argument('project',help='The project name')
        subcommand.add_argument('-n','--name',help='The model name')
        subcommand.add_argument('-d','--directory',help='The directory where to store the project.',default='.')
        subcommand.add_argument('-r','--readme',help='The description of the project.')

    def git(self):
        '''
        Subcommand git, which adds a git repository to this project.
        '''
        subcommand=self.add_subcommand('git',git)

    def add(self):
        '''
        Subcommand add, which adds an engine to this project.
        '''
        subcommand=self.add_subcommand('add',add)
        subcommand.add_argument('engine',help='The engine to be added to the project.',choices=['tba','ed','vca','dmrg'])

    def execute(self):
        '''
        Execute the command line commands this manager manages.
        '''
        namespace=self.parser.parse_args(self.args[1:] or ['-h'])
        namespace.subcommand(**{key:value for key,value in vars(namespace).iteritems() if key!='subcommand'})

def init(directory,project,name=None,readme=None):
    '''
    Initialize a project.

    Parameters
    ----------
    directory : str
        The directory where to store the project.
    project : str
        The project name.
    name : str, optional
        The model name.
    readme : str, optional
        The description of the project.
    '''
    pdir='%s/%s'%(directory,project)
    ddir,rdir,sdir='%s/data'%pdir,'%s/result'%pdir,'%s/source'%pdir
    dirs=[pdir,ddir,rdir,sdir]
    for folder in dirs:
        if not os.path.exists(folder): os.makedirs(folder)
    with open('%s/README.md'%pdir,'w') as fout:
        fout.write('# %s'%project)
        fout.write('\n')
        if readme:
            fout.write(readme)
            fout.write('\n')
    with open('%s/manager.py'%pdir,'w') as fout:
        for line in Template.manager():
            fout.write(line)
            fout.write('\n')
    with open('%s/basics.py'%sdir,'w') as fout:
        for line in Template.basics(name or project):
            fout.write(line)
            fout.write('\n')
    with open('%s/__init__.py'%sdir,'w') as fout:
        fout.write('from basics import *')
        fout.write('\n')

def git(args=None):
    pass

def add(engine):
    pass
