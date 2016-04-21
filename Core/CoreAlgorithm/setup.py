def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config=Configuration('CoreAlgorithm',parent_package,top_path)
    #config.add_extension('VCA_Fortran',['VCA_Fortran.f90'])
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
