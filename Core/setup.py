def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config=Configuration('Core',parent_package,top_path)
    #config.add_subpackage('BasicAlgorithm')
    config.add_subpackage('BasicClass')
    #config.add_subpackage('CoreAlgorithm')
    #config.add_subpackage('Test')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
