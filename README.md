# lattice-boltzmann-gpu
Lattice Boltzmann Method (LBM) code with GPU offloading via OpenMP

This is a simple Lattice Boltzmann Method CFD code that offloads computations to GPU via OpenMP. 
It solves the 2D flow of two tandem cylinders by using the D2Q9 lattice model with Single-Relaxation-Time (SRT). 
The use of directive-based OpenMP API in the code allows portability to any supported accelerator. 
The main objective of this project is to serve as an open-source reference for GPU offloading implementation when developing Lattice Boltzmann Method.

 ![U_field](https://github.com/francescoornano/lattice-boltzmann-gpu/assets/102365359/e3bf445f-61e6-4e92-85eb-e0c05b500986)

In order to compile the code, please download the NVIDIA HPC Software Development Kit at: 
https://developer.nvidia.com/hpc-sdk

If you are using WSL2, you need to add the following lines to your bashrc file (check the installed SDK version):
```
export PATH="$PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin/"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib/" 
```

To compile and run the code open a terminal and run:
```
sh run.sh
```

The run.sh file will compile uisng nvc++ compiler and offload to an available device (-mp flag is set to -mp=gpu). 
The code will output both x and y velocity components in .csv format.
