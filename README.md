# 6-881-examples
[![Build Status](https://travis-ci.org/RobotLocomotion/6-881-examples.svg?branch=master)](https://travis-ci.org/RobotLocomotion/6-881-examples)

This repository contains a collection of tools for interacting with robots and cameras, developed to support the Intelligent Robot Manipulation class [(MIT 6.881)](https://manipulation.csail.mit.edu/).

## Pre-reqs
In the root directory of this repository, run the following command in a terminal to build a docker image that includes Drake and denpendencies for PDDLStream:
```bash
$ docker build -t mit6881 -f ubuntu16_04_mit6881.dockerfile --build-arg DRAKE_VERSION=20181211 .
``` 

## Use
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





