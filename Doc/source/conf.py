import sys
import os

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
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
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True

# sphinx settings
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
autoclass_content = 'both'

# General information about the project.
project = u'HamiltonianPy'
copyright = u'2017, Zhao-Long Gu'
author = u'Zhao-Long Gu'
version = u'1.2'
release = u'1.2.0'

# other settings
language = None
exclude_patterns = []
pygments_style = 'sphinx'

# html settings
#html_theme = 'nature'
html_theme = 'classic'
html_static_path = ['_static']
htmlhelp_basename = 'HamiltonianPydoc'

# latex settings
latex_elements = {}
latex_documents = [
    (master_doc, 'HamiltonianPy.tex', u'HamiltonianPy Documentation',
     u'Zhao-Long Gu', 'manual'),
]

# man settings
man_pages = [
    (master_doc, 'hamiltonianpy', u'HamiltonianPy Documentation',
     [author], 1)
]

texinfo_documents = [
    (master_doc, 'HamiltonianPy', u'HamiltonianPy Documentation',
     author, 'HamiltonianPy', 'One line description of project.',
     'Miscellaneous'),
]

intersphinx_mapping = {'https://docs.python.org/': None}
