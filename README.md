# 6-881-examples

This repository contains a collection of tools for interacting with robots and cameras, developed to support the Intelligent Robot Manipulation class [(MIT 6.881)](https://manipulation.csail.mit.edu/) and as a part of a Master's thesis. This code has only been tested on Ubuntu 16.04 with Drake versions before June 1, 2019. Note that there are also experimental dependencies that do not live in the master branch of Drake.

Most of this code can be run in a docker container, however perception tools relating to [NVidia's DOPE](https://github.com/NVlabs/Deep_Object_Pose) can only be run on a machine that can install and run pytorch.

## Docker Use

In the root directory of this repository, run the following command in a terminal to build a docker image that includes Drake and denpendencies for PDDLStream:
```bash
$ docker build -t mit6881 -f ubuntu16_04_mit6881.dockerfile --build-arg DRAKE_VERSION=20181212 .
``` 

In the root directory of this repository, run 
```bash
$ python ./docker_run.py --os [your operating system]
``` 
where `[your operating system]` should be replaced with `mac` or `linux`. This command will start a docker container (virtual machine) with the docker image you have created. The `6-881-examples` folder on the host machine (your laptop/desktop) is mounted to `/6-881-examples` in the docker container. 

In the docker container, run
```bash
$ terminator
```
to launch `terminator`, a popular terminal multiplexer on linux. The terminator window is launched with a dark green background to distinct itself from terminals running on the host machine. 

## Local Use

In order to use DOPE, all packages must be installed on your machine. This has only been partially tested on Ubuntu 16.04, however these exact instructions have not been fully tested. If any dependencies are missing or some of these commands don't work, please file an Issue describing the problem.

### Dependencies
The following Python 2 packages are requried to run this code:

* graphviz
* numpy
* meshcat 
* jupyter 
* timeout-decorator 
* sklearn 
* scipy 
* py-trees version 0.5.9
* PIL
* yaml
* cv2
* pyrr
* enum
* json
* threading
* torch
* matplotlib

A sourced version of Drake is needed, along with its Python 2 bindings. This code relies on an experimental version of Pang Tao's Drake. To install that version and build its Pythonn bindings, run the following commands.

```
$ git clone -b robot_plan_runner --single-branch https://github.com/pangtao22/drake.git drake
$ cd drake && yes "Y" | ./setup/ubuntu/install_prereqs.sh && cd ..
$ mkdir drake-build \
    && cd drake-build \
    && cmake ../drake \
    && make -j
$ export PYTHONPATH=${PWD}/install/lib/python2.7/site-packages:${PYTHONPATH}
```

More information about [installing  a sourced version of Drake](https://drake.mit.edu/from_source.html) and [building its Python bindings](https://drake.mit.edu/python_bindings.html#building-the-python-bindings) can be found on the main Drake website.
