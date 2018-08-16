def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config=Configuration('TensorNetwork',parent_package,top_path)
    config.add_subpackage('Tensor')
    config.add_subpackage('Structure')
    config.add_subpackage('MPS')
    config.add_subpackage('MPO')
    config.add_subpackage('test')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
