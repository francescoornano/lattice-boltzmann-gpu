#!bin/bash

rm u_*
nvc++ Main.cpp -o Main_nvc++ -Minfo=mp -mp=gpu -Ofast
export OMP_TARGET_OFFLOAD=MANDATORY
./Main_nvc++

