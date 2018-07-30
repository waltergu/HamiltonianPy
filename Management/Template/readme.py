'''
readme template.
'''

__all__=['readme']

readme_template="""\
# {project}
{description}

Authors
-------
{authors}

Contact
-------
{email}
"""

def readme(project,description,authors,email):
    return readme_template.format(
            project=        project,
            description=    description,
            authors=        '\n'.join('* %s'%author.lstrip() for author in authors.split(',')),
            email=          email
            )