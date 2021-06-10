# Steps to run our reproducible report:

1. Install python dependencies:
    pip install -r requirements.txt
    
2. Install jupyter notebook dependencies (for readability):
    Instructions are from https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html
    2.1. pip install jupyter_contrib_nbextensions
    2.2. jupyter contrib nbextension install --user
    Enable codefolding, table of contents and pythom-markdown extensions
    2.3. jupyter nbextension enable codefolding/main
    2.4. jupyter nbextension enable toc2/main
    2.5. jupyter nbextension enable python-markdown/main

3. Run notebook
    3.1. Navigate to 'Reproducible Report.ipynb' in Jupyter
    3.2. Change the file paths in the code cell underneath section 1.1 of the report to point to the specified directories
    3.3. Run the notebook
