'''
gitignore template.
'''

__all__=['gitignore']

gitignore_template="""\
# python
*.py[cod]
build/
data/
log/

# pycharm
.idea/

# vscode
.vscode/

# tex
*.aux
*.bak
*.bbl
*.out
*.sav
*.gz
*.rar
*.log
*.blg
*Notes.bib
"""

def gitignore():
    return gitignore_template
