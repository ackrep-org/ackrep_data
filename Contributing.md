# Preliminary Guide for Contributing to this Repo

- Preparation:
    - Install the ackrep software:
        - clone the [ackrep_core](https://github.com/cknoll/ackrep_core) repo
        - run `pip install -e .` in the root directory of that repo
        - confirm installation: the command `ackrep --help` should print some help text of the ackrep command line utility
    - Clone [ackrep_data](https://github.com/cknoll/ackrep_data) repo
- Integration of new data:
    - New `ProblemSpecification`:
        - Create a new subirectory: like `ackrep_data/problem_specifications/<your_problem>`.
        - Copy your problem-specification-file into the directory and change name to `problem.py`.
        - Generate a new random key: `ackrep --key`.
        - Copy the file `metadata.yml` from an existing entity of the same type and change its content beginning with the new key.
        - Ensure that the function `evaluate_solution` returns an instance of `ackrep_core.ResultContainer` see examples.
    - New `MethodPackage`:
        - Create a new subirectory: like `ackrep_data/method_packages/<your_method>/src`.
        - Copy your code into that directory.
        - Generate a new random key: `ackrep --key`.
        - Copy the file `metadata.yml` from an existing entity of the same type to `ackrep_data/method_packages/<your_method>` and change its content beginning with the new key.
    - New `ProblemSolution`:
        - Create a new subirectory: like `ackrep_data/problem_solutions/<your_problem_solution>`.
        - Copy your problem-solution-file into the directory and change name to `solution.py`.
        - Generate a new random key: `ackrep --key`.
        - Copy the file `metadata.yml` from an existing entity of the same type and change its content beginning with the new key.
        - Add all keys of solved problems and dependent packages.
- Evaluate solution:
    - Change working directory to `ackrep_data`.
    - Run `ackrep --load-repo-to-db .` in that directory.
    - Check a specific solution via `ackrep --check-solution problem_solutions/<your_problem_solution>/metadata.yml`.
    - If an error occurs:
        - Manually run `python execscript.py` and see the error messages.
        - Common issues:
            - Forgot to run `ackrep --load-repo-to-db .` after changing metadata.
            - ImportError due to missing package in the python_path (see `execscript.py` what actually is inserted); Probable cause: missing or wrong key in `metadata.yml`
            - Bad return-value of `evaluate_solution`
