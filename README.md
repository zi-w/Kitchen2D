# Kitchen2D

This is a code repository of learning for task and motion planning in a 2D kitchen. 
This repository is associated with the paper [_Active model learning and diverse action sampling for task and motion planning_](https://arxiv.org/abs/1803.00967). Our project page is [here](https://ziw.mit.edu/projects/kitchen2d/). If you have any question, please email me at ziw 'at' mit.edu, or create an issue [here](https://github.com/zi-w/Kitchen2D/issues).

We developed our code building upon several existing packages:
* [pybox2d](https://github.com/pybox2d/pybox2d), a 2D physics engine for Python based on the [Box2D](http://box2d.org/) library;
* [GPy](https://sheffieldml.github.io/GPy/), a Gaussian process framework in Python;
* [motion-planners](https://github.com/caelan/motion-planners), including basic motion planning functionals such as rapidly-exploring random trees;
* [pddlstream](https://github.com/caelan/pddlstream), lightweight implementation of [STRIPStream](https://github.com/caelan/stripstream), which builds upon the [Fast Downward](http://www.fast-downward.org) planner.
* [numpy](http://www.numpy.org/), version 1.13.3 or higher.
* [scipy](https://www.scipy.org), version 0.19.1 or higher.
* [sklearn](http://scikit-learn.org/stable/), version 0.18.1 or higher.

In particular, motion-planners and pddlstream are included as submodules in this repository. 

[![Video Demo](https://img.youtube.com/vi/QWjLYjN8axg/0.jpg)](https://www.youtube.com/watch?v=QWjLYjN8axg)

## System Requirement
We tested our code with Python 2.7.6 on Ubuntu 14.04 LTS (64-bit) and Mac OS X. To install pybox2d, GPy and Fast Downward, follow the following steps.

1. Install numpy, scipy and sklearn, following the instructions [here](https://www.scipy.org/install.html) and [here](http://scikit-learn.org/stable/install.html).

2. Follow the instructions [here](https://github.com/pybox2d/pybox2d/blob/master/INSTALL.md) to install [pybox2d](https://github.com/pybox2d/pybox2d).

3. To run the learning examples, follow the instructions [here](https://github.com/SheffieldML/GPy#getting-started-installing-with-pip) to install [GPy](https://sheffieldml.github.io/GPy/).

4. To run the planning examples, follow the instructions [here](http://www.fast-downward.org/ObtainingAndRunningFastDownward) to obtain [Fast Downward](http://www.fast-downward.org).

## Quick Start
Once you confirm the system requirements are satisfied, make a copy of this repository with your favorite method, e.g. 
```
git clone git@github.com:YOUR_USERNAME/Kitchen2D.git
cd Kitchen2D
```
Initialize and update the submodules by
```
git submodule init
git submodule update
```
Now you should be able to run the examples below.

## Examples

### Example of Primitives
An example of using the primitives is in primitive_example.py. Try
```
python primitive_example.py
```

### Example of Learning
We show an example of both learning and sampling the scooping action in learn_example.py. We adopt an active learning view to learn the feasible region of the pre-conditions of the primitives. In order to plan with the learned pre-conditions, we need to be able to sample from its feasible regions. The detailed algorithm and setups we used can be found in [Section IV.A of the accompanying paper](https://arxiv.org/abs/1803.00967). Try
```
python learn_example.py
```

### Example of Planning
plan_example.py is an example of planning with learned pouring and scooping actions. We use STRIPSream as the backend planner. The goal of the task in plan_example.py is to “serve” a cup of coffee with cream and sugar by placing it on the green coaster near the edge of the table. [Click here for vidoes of plans](https://www.youtube.com/playlist?list=PLoWhBFPMfSzDbc8CYelsbHZa1d3uz-W_c&disable_polymer=true). Try
```
python plan_example.py
```


## Citation
Please cite our work if you would like to use the code.
```
@inproceedings{wangIROS2018,
    author={Zi Wang and Caelan Reed Garrett and Leslie Pack Kaelbling and Tomas Lozano-Perez},
    title={Active model learning and diverse action sampling for task and motion planning},
    booktitle={International Conference on Intelligent Robots and Systems (IROS)},
    year={2018},
    url={http://lis.csail.mit.edu/pubs/wang-iros18.pdf}
}
```

## References
* Active model learning and diverse action sampling for task and motion planning (Zi Wang, Caelan Reed Garrett, Leslie Pack Kaelbling and Tomas Lozano-Perez), In International Conference on Intelligent Robots and Systems (IROS), 2018.
