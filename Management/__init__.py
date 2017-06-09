'''
This subpackage implements the construction and maintenance of a project through command line commands, including:
    * classes: Manager
    * functions: init, add
'''

import os
import Template
from argparse import ArgumentParser

__all__=['Manager','init','add']

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
        subcommand.add_argument('-a','--authors',help='The authors, seperated by commas, of the project.',default='Anonymous')
        subcommand.add_argument('-e','--email',help='The contact email.',default='.')
        subcommand.add_argument('-r','--readme',help='The description of the project.',default='')
        subcommand.add_argument('-l','--license',help="'T'/'t' for adding a GNU GPL v3.0 LICENSE file and 'F'/'f' for not.",default='t',choices=['T','t','F','f'])
        subcommand.add_argument('-g','--gitignore',help="'T'/'t' for adding a gitignore file and 'F'/'f' for not.",default='t',choices=['T','t','F','f'])

    def add(self):
        '''
        Subcommand add, which adds an engine to this project.
        '''
        subcommand=self.add_subcommand('add',add)
        subcommand.add_argument('engine',help='The engine to be added to the project.',choices=['tba','ed','vca','dmrg'])
        subcommand.add_argument('-s','--system',help="'spin' for spin systems and 'fermi' for fermionic systems.",default='fermi',choices=['spin','fermi'])

    def execute(self):
        '''
        Execute the command line commands this manager manages.
        '''
        namespace=self.parser.parse_args(self.args[1:] or ['-h'])
        namespace.subcommand(**{key:value for key,value in vars(namespace).iteritems() if key!='subcommand'})

    @classmethod
    def has_project(self):
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
        The authors, seperated by commas, of the project.
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
        fout.write('# %s'%project)
        fout.write('\n')
        if readme:
            fout.write(readme)
            fout.write('\n')
        if authors!='Anonymous':
            fout.write('\nAuthors\n')
            fout.write('-------\n')
            fout.write('\n'.join('* %s'%author.lstrip() for author in authors.split(',')))
            fout.write('\n')
        if email:
            fout.write('\nContact\n')
            fout.write('-------\n')
            fout.write('%s\n'%email)
            fout.write('\n')
    if license.upper()=='T':
        with open('%s/LICENSE'%pdir,'w') as fout:
            for line in Template.license(authors):
                if line is not None:
                    fout.write(line)
                    fout.write('\n')
    if gitignore.upper()=='T':
        with open('%s/.gitignore'%pdir,'w') as fout:
            for line in Template.gitignore():
                if line is not None:
                    fout.write(line)
                    fout.write('\n')
    with open('%s/manager.py'%pdir,'w') as fout:
        for line in Template.manager():
            if line is not None:
                fout.write(line)
                fout.write('\n')
    with open('%s/config.py'%sdir,'w') as fout:
        for line in Template.config():
            if line is not None:
                fout.write(line)
                fout.write('\n')
    with open('%s/__init__.py'%sdir,'w') as fout:
        fout.write('from config import *')
        fout.write('\n')

def add(engine,system):
    '''
    Add an engine to a project.

    Parameters
    ----------
    engine : 'tba','ed','vca','dmrg'
        The engine ot be added.
    system : 'spin' or 'fermi'
        'spin' for spin systems and 'fermi' for fermionic systems.
    '''
    if Manager.has_project():
        if not os.path.exists('log/%s'%engine): os.makedirs('log/%s'%engine)
        if not os.path.exists('data/%s'%engine): os.makedirs('data/%s'%engine)
        if not os.path.exists('result/%s'%engine): os.makedirs('result/%s'%engine)
        with open('source/%s.py'%engine,'w') as fout:
            for line in getattr(Template,engine)(system):
                if line is not None:
                    fout.write(line)
                    fout.write('\n')
        with open('source/__init__.py') as fin:
            content=set(fin.readlines())
        with open('source/__init__.py','a+') as fout:
            line='from %s import *\n'%engine
            if line not in content:
                fout.write(line)
    else:
        raise ValueError('add error: No project found.')
