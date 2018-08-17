'''
This subpackage implements the construction and maintenance of a project through command line commands, including:
    * classes: Manager
    * functions: init, add
'''

import os
from . import Template
from argparse import ArgumentParser

__all__=['Manager','init','add']

class Manager(object):
    '''
    The manager of project construction and maintenance.

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
        subcommand.add_argument('-d','--directory',help='The directory where to store the project.',default='.')
        subcommand.add_argument('-a','--authors',help='The authors, separated by commas, of the project.',default='Anonymous')
        subcommand.add_argument('-e','--email',help='The contact email.',default='')
        subcommand.add_argument('-r','--readme',help='The description of the project.',default='')
        subcommand.add_argument('-l','--license',help="'T'/'t' for adding a GNU GPL v3.0 LICENSE file and 'F'/'f' for not.",default='t',choices=['T','t','F','f'])
        subcommand.add_argument('-g','--gitignore',help="'T'/'t' for adding a gitignore file and 'F'/'f' for not.",default='t',choices=['T','t','F','f'])

    def add(self):
        '''
        Subcommand add, which adds an engine to this project.
        '''
        subcommand=self.add_subcommand('add',add)
        subcommand.add_argument('engine',help='The engine to be added to the project.',choices=['tba','ed','vca','vcacct','idmrg','fdmrg','fbfm'])
        subcommand.add_argument('-s','--system',help="'spin' for spin systems and 'fock' for fock systems.",default='fock',choices=['spin','fock'])

    def execute(self):
        '''
        Execute the command line commands this manager manages.
        '''
        namespace=self.parser.parse_args(self.args[1:] or ['-h'])
        namespace.subcommand(**{key:value for key,value in vars(namespace).items() if key!='subcommand'})

    @classmethod
    def hasproject(cls):
        '''
        Judge whether the current folder contains a project.
        '''
        try:
            with open('README.md','r') as fin:
                name=fin.readline()[2:-1]
            assert name==os.path.basename(os.getcwd())
            return True
        except:
            return False

def init(directory,project,authors,email,readme,license,gitignore):
    '''
    Initialize a project.

    Parameters
    ----------
    directory : str
        The directory where to store the project.
    project : str
        The project name.
    authors : str
        The authors, separated by commas, of the project.
    email : str
        The contact email.
    readme : str
        The description of the project.
    license : 'T','t','F','f'
        'T'/'t' for adding a GNU GPL v3.0 LICENSE file and 'F'/'f' for not.
    gitignore : 'T','t','F','f'
        'T'/'t' for adding a gitignore file and 'F'/'f' for not.
    '''
    pdir='%s/%s'%(directory,project)
    ddir,rdir,sdir,ldir='%s/data'%pdir,'%s/result'%pdir,'%s/source'%pdir,'%s/log'%pdir
    dirs=[pdir,ddir,rdir,sdir,ldir]
    for folder in dirs:
        if not os.path.exists(folder): os.makedirs(folder)
    with open('%s/README.md'%pdir,'w') as fout:
        fout.write(Template.readme(project,readme,authors,email))
    if license.upper()=='T':
        with open('%s/LICENSE'%pdir,'w') as fout:
            fout.write(Template.license(authors))
    if gitignore.upper()=='T':
        with open('%s/.gitignore'%pdir,'w') as fout:
            fout.write(Template.gitignore())
    with open('%s/manager.py'%pdir,'w') as fout:
        fout.write(Template.manager())
    with open('%s/config.py'%sdir,'w') as fout:
        fout.write(Template.config())
    with open('%s/__init__.py'%sdir,'w') as fout:
        fout.write('from .config import *\n')

def add(engine,**kargs):
    '''
    Add an engine to a project.

    Parameters
    ----------
    engine : 'tba','ed','vca','vcacct','fdmrg','idmrg','fbfm'
        The engine ot be added.
    '''
    if Manager.hasproject():
        if not os.path.exists('result/%s'%engine):
            os.makedirs('result/%s'%engine)
        with open('source/%s.py'%engine,'w') as fout:
            fout.write(getattr(Template,engine)(**kargs))
        with open('source/__init__.py') as fin:
            content=set(fin.readlines())
        with open('source/__init__.py','a+') as fout:
            line='from .%s import *\n'%engine
            if line not in content:
                fout.write(line)
    else:
        raise ValueError('add error: No project found.')
