This program is used to simulate the positron decay of the source and the transportation of the gammas in the phantom
The positron transportation part is under construction. Currently the gamma pair is assumed to be produced in where positron is emitted.

To compile the files, simply use make command and an executable file name gbrachy_test will be produced

The files that is needed to run the simulation are
1, Voxelized phantom, including density file and material file. See ./patientcase/waterPhantom_den.bin and ./patientcase/waterPhantom_mat.bin, for example. 
The first three lines are Xbins, Ybins, Zbins; Xoffset, Yoffset, Zoffset; Xdim, Ydim, Zdim. The following lines are the corresponding data.
2,  Voxelized source, see ./patientcase/S_listReal.bin. In each lines, the weight of different source is defined.
3, input file input_PET.in. In the input file, every two rows are combined to give an input for a parameter. The first one is the describing one and the other gives the value.

To run the simulation, use command 
./gbrachy_test < input_PET.in

When the simulation is finished, one PSF file named phaseSpace.dat will be generated that can be used in another detector program.