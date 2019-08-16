
Note: the project is in the early stage of a feasibility study.

# Automatic Control Knowledge Repository (ACKRep) Data

This repository holds the data for the ACKRep project. Due to the subject and the concept of
that project, data consists mainly in source code and metadata.

# Basic Concept of ACKRep


The project aims to represent knowledge (in the field of automatic control) in a special way:
as a combination of *problem-specifications* and *problem-solutions*. Both, problems and solutions
can be represented by software modules. To ensure integrity and reproducability,
solution-modules are checked automatically against the respective problems. To facilitate the
operation of this process, some metadata and some other components are also managed in this repo.
Alltogether there are the following entities:

- problem-class
    - e.g. Trajectory Planning for Mechanical Systems
- problem-specification
    - e.g. Swingup control for the Acrobot with specific parameters
- problem-solution
    - e.g. Invocation of an allgorithm which calculates a swingup trajectory for the acrobot
- method
    - e.g. A collection of functions which together implement an algorithm to calculate state space
    transitions for dynamical systems
- docs
    - e.g. A mathematical description of that algorithm e.g. as `.tex`-file
- environment specification
    - A set of rules to setup a software environment such that the above software components
    can be executed
- comments
    - A structured way to add remarks or questions to an entity

**Summary**: In some sense ACKrep translates the concept of continous integration from software
engineering to research in automatic control.

# License

This repo might contain source code among different free software licenses. Potential contributors
are encouraged to choose GPLv3. Alternatively MIT license would also be OK. For any file in this
repo, the license is applicable (in this order) which noted inside the file itself, in a
`LICENSE`-file inside the same directory or in the nearest parent directory containing a
`LICENSE`-file. As the root directory contains a `LICENSE`-file with GPLv3, this is the default
for all subdirectories.


