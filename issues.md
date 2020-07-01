This file is a collection of issues, ideas and questions for the maintainer. Currently, this is more effective to handle than the github-issues.


- Handle jupyter-notebook files as solutions e.g. (`solution.ipynb`, `postprocessing.ipynb`)?
    - (+) more expressive, "executable documentation"
    - (-) we cannot directly import from a notebook
    - **temporary decision**: for now we do not support jupyter-notebooks
- Can we specify our data models like a django models? This would make consistency checks simple
- add email adresses in this format: <firstname.lastname(et)provider(dot)org>.
- every object should have an base_path attribute

