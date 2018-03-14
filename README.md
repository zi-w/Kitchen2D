# Kitchen2D (Under construction)

This is a code repository of learning for task and motion planning in a 2D kitchen. 
This repository is associated with the paper [_Active model learning and diverse action sampling for task and motion planning_](https://arxiv.org/abs/1803.00967). Our project page is [here](http://zi-wang.com/kitchen2d/). If you have any question, please email me at ziw 'at' mit.edu, or create an issue [here](https://github.com/zi-w/Kitchen2D/issues).

We developed our code building upon several existing packages:
* [pybox2d](https://github.com/pybox2d/pybox2d), a 2D physics engine for Python based on the [Box2D](http://box2d.org/) library;
* [GPy](https://sheffieldml.github.io/GPy/), a Gaussian process framework in Python;
* [motion-planners](https://github.com/caelan/motion-planners), including basic motion planning functionals such as rapidly-exploring random trees;
* [STRIPStream Light](https://github.mit.edu/caelan/ss), lightweight implementation of [STRIPStream](https://github.com/caelan/stripstream), which builds upon the [Fast Downward](http://www.fast-downward.org) planner.

In particular, motion-planners and STRIPStream Light are included in this repository but you will need to install pybox2d, GPy and Fast Downward to run the examples. 

## System Requirement
We tested our code with Python 2.7.6 on Ubuntu 14.04 LTS (64-bit) and Mac OS X. To install pybox2d, GPy and Fast Downward, follow the following steps.

1. Follow the instructions [here](https://github.com/pybox2d/pybox2d/blob/master/INSTALL.md) to install [pybox2d](https://github.com/pybox2d/pybox2d).

2. To run the learning examples, follow the instructions [here](https://github.com/SheffieldML/GPy#getting-started-installing-with-pip) to install [GPy](https://sheffieldml.github.io/GPy/).

3. To run the planning examples, follow the instructions [here](http://www.fast-downward.org/ObtainingAndRunningFastDownward) to obtain [Fast Downward](http://www.fast-downward.org).

## Examples

### Example of Primitives
The motion premitives are in kitchen_stuff.py. An example of using the primitives is in primitive_example.py.

### Example of Learning
We show an example of both learning and sampling the scooping action in learn_example.py. We adopt an active learning view to learn the feasible region of the pre-conditions of the primitives. In order to plan with the learned pre-conditions, we need to be able to sample from its feasible regions. The detailed algorithm and setups we used can be found in [Section IV.A of the accompanying paper](https://arxiv.org/abs/1803.00967).

### Example of Planning
coffee_task.py is an example of planning with learned pouring and scooping actions. We use STRIPSream as the backend planner. The goal is to “serve” a cup of coffee with cream and sugar by placing it on the green coaster near the edge of the table. [Click here for vidoes of plans.](https://www.youtube.com/playlist?list=PLoWhBFPMfSzDbc8CYelsbHZa1d3uz-W_c&disable_polymer=true) 


## Citation
Please cite our work if you would like to use the code.
```
@article{wang2018active,
  title={Active model learning and diverse action sampling for task and motion planning},
  author={Wang, Zi and Garrett, Caelan Reed and Kaelbling, Leslie Pack and Lozano-P{\'e}rez, Tom{\'a}s},
  journal={arXiv preprint arXiv:1803.00967},
  year={2018}
}
```

## References
* Wang Z, Garrett CR, Kaelbling LP, Lozano-Pérez T. Active model learning and diverse action sampling for task and motion planning. arXiv preprint arXiv:1803.00967. 2018 Mar 2.