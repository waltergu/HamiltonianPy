# HamiltonianPy documentation build configuration file

# Sphinx extension module
extensions=[
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

# Napoleon settings
napoleon_google_docstring=False
napoleon_numpy_docstring=True
napoleon_include_init_with_doc=False
napoleon_include_private_with_doc=True
napoleon_include_special_with_doc=True
napoleon_use_admonition_for_examples=False
napoleon_use_admonition_for_notes=True
napoleon_use_admonition_for_references=True
napoleon_use_ivar=True
napoleon_use_param=True
napoleon_use_rtype=True
napoleon_use_keyword=True

# autodoc settings
autoclass_content='both'
source_suffix='.rst'
master_doc='index'

# General information about the project.
project=u'HamiltonianPy'
copyright=u'2017, Zhao-Long Gu'
author=u'Zhao-Long Gu'
version=u'1.2.0'
release=u'1.2.0'

# Options for HTML output
html_theme='classic'
html_last_updated_fmt=''
html_sidebars={'**':['globaltoc.html','relations.html','sourcelink.html','searchbox.html']}