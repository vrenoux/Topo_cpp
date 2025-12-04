#!/bin/bash
g++ -fdiagnostics-color=always -g main.cpp src/*.cpp -o main \
$(pkg-config --cflags --libs petsc) \
-I/usr/lib/x86_64-linux-gnu/openmpi/include \
-lmpi

