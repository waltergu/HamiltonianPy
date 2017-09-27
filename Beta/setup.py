def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config=Configuration('Beta',parent_package,top_path)
    config.add_subpackage('TrED')
    config.add_subpackage('QMC')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
