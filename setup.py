def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config=Configuration('HamiltonianPy',parent_package,top_path,version='0.0.0',author='waltergu',author_email='waltergu@126.com')
    config.add_subpackage('Math')
    config.add_subpackage('Basics')
    config.add_subpackage('DataBase')
    config.add_subpackage('FreeSystem')
    config.add_subpackage('ED')
    config.add_subpackage('VCA')
    config.add_subpackage('MERA')
    config.add_subpackage('Test')
    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
