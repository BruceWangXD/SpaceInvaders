# Steps to read our reproducible report:
1. Navigate to index.html which can be found in Report/report_book/_build/html, and open it in browser (Chrome recommended).


# Steps to run our reproducible report:
1. Navigate to Report/requirements/

2. Install python dependencies:
    pip install -r requirements.txt
    
3. Install jupyter notebook dependencies (for readability):
    Instructions are from https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html
    3.1. pip install jupyter_contrib_nbextensions
    3.2. jupyter contrib nbextension install --user
    Enable codefolding, table of contents and pythom-markdown extensions
    3.3. jupyter nbextension enable codefolding/main
    3.4. jupyter nbextension enable toc2/main
    3.5. jupyter nbextension enable python-markdown/main

4. Run notebook
    4.1. Navigate to 'Report/report_book/Reproducible_Report.ipynb' in Jupyter
    4.2. Change the file paths and desired options in the code cell underneath Dependencies sections
    4.3. Run the notebook
