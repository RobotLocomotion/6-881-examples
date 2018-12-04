#!/usr/bin/env python
from __future__ import print_function

import argparse
import os
import subprocess
import getpass

if __name__=="__main__":
    user_name = getpass.getuser()
    parser = argparse.ArgumentParser()

    parser.add_argument("--os", type=str, choices=['mac', 'linux'], required=True, help='operating system of the host machine.')

    parser.add_argument("-c", "--container", type=str, default="6-881-examples", help="(optional) name of the container")

    parser.add_argument("-d", "--dry_run", action='store_true', help="(optional) perform a dry_run, print the command that would have been executed but don't execute it.")

    parser.add_argument("-e", "--entrypoint", type=str, default="/bin/bash", help="(optional) thing to run in container")

    parser.add_argument("-p", "--passthrough", type=str, default="", help="(optional) extra string that will be tacked onto the docker run command, allows you to pass extra options. Make sure to put this in quotes and leave a space before the first character")

    args = parser.parse_args()
    source_dir = os.getcwd()

    image_name = 'mit6881'
    print("running docker container derived from image %s" % image_name)
    home_directory = '/home/' + user_name

    if args.os == 'linux':
        cmd = "xhost +local:root \n"
    elif args.os == 'mac':
        IP = subprocess.check_output("echo $(ifconfig en0 | grep inet | awk '$1==\"inet\" {print $2}')", shell=True)[:-1]
        cmd = "/usr/X11/bin/xhost + " + IP + "\n"
    
    cmd += "docker run "
    if args.container:
        cmd += " --name %(container_name)s " % {'container_name': args.container}

    # enable graphics
    if args.os == 'linux':
        cmd += " -e DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw "
    elif args.os == "mac":
        cmd += " -e DISPLAY=" + IP + ":0 "     

    cmd += " -v ~/.ssh:%(home_directory)s/.ssh " % {'home_directory': home_directory}   # mount ssh keys
    cmd += " -v '%(source_dir)s':/6-881-examples/ " % {'source_dir': source_dir}

    # network settings
    if args.os == 'mac':
        cmd += " -p 7000-7010:7000-7010 " # meshcat port
        cmd += " -p 8080:8080 " # jupyter notebook port
    elif args.os == 'linux':
        cmd += " --network=host " # use host machine's network. 

    cmd += " " + args.passthrough + " "

    cmd += " --rm "  # remove the container when you exit
    cmd += " --ipc=host "

    cmd += " --entrypoint " + args.entrypoint
    cmd += " -it "
    cmd += image_name

    if args.os == "linux":
        cmd_endxhost = "xhost -local:root"
    elif args.os == "mac":
        cmd_endxhost = "/usr/X11/bin/xhost - " + IP
    print("command = \n \n", cmd, "\n", cmd_endxhost)
    print("")

    # run the docker image
    if not args.dry_run:
        print("executing shell command")
        code = os.system(cmd)
        print("Executed with code ", code)
        os.system(cmd_endxhost)
        # Squash return code to 0/1, as
        # Docker's very large return codes
        # were tricking Jenkins' failure
        # detection
        exit(code != 0)
    else:
        print("dry run, not executing command")
        exit(0)
