# gPET
GPU version of the PET simualtion
This is the Version 1.0, which can simulate the decay of the source, the transportation in the phantom and the detector and finally the signals of PET.

Please first run the initialize.sh to compile all the source codes. The CUDA is required before you compile. You may need to revise the Makefile to make the path to the CUDA correct. 
CUDA_INSTALL_PATH := Your path to cuda installation

After the initialization, two exceutable file named gbranchy_test and gPET_test will appear in the two seperate child folders, respectively. Then simply run the run.sh to get the results for the example input. The two program can be run seperately, for which you can refer to their readme.txt file in corresponding folder.

The simulation is controled by the two input_PET.in files, which give the simulation parameters. Try to revise them to your own geometry and data.

Enjoy!
01-20-2019
