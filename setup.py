def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from platform import system
    config=Configuration('HamiltonianPy',parent_package,top_path,version='0.0.0',author='waltergu',author_email='waltergu@126.com')
    config.add_subpackage('Misc')
    config.add_subpackage('Basics')
    config.add_subpackage('TensorNetwork')
    config.add_subpackage('FreeSystem')
    config.add_subpackage('FBFM')
    config.add_subpackage('ED')
    config.add_subpackage('VCA')
    config.add_subpackage('DMRG')
    config.add_subpackage('Management')
    config.add_subpackage('Test')
    config.add_subpackage('Beta')
    config.add_scripts('Hamiltonian')
    if system()=='Windows': config.add_scripts('Hamiltonian.bat')
    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
