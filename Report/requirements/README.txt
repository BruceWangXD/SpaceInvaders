# Steps to run our reproducible report:

1. Install python dependencies:
    pip install -r requirements.txt
    
2. Install jupyter notebook dependencies (for readability):
    Instructions are from https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html
    2.1. pip install jupyter_contrib_nbextensions
    2.2. jupyter contrib nbextension install --user
    Enable codefolding and table of contents extensions
    2.3. jupyter nbextension enable codefolding/main
    2.4. jupyter nbextension enable toc2/main

3. Run notebook
    Navigate to 'Reproducible Report.ipynb' in Jupyter and run the notebook.
