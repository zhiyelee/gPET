This program is used to simulate the transportation of the gammas in the detector and produce signals

To compile the files, simply use make command and an executable file name gPET_test will be produced

The files that is needed to run the simulation are
1, Parametrized detector. See ./patientcase/PETconfiguration/config_ro.geo, for example. 
2, input file input_PET.in. In the input file, every two rows are combined to give an input for a parameter. The first one is the describing one and the other gives the value.
3, PSF file, the default one is ./patientcase/PETconfiguration/phaseSpace.dat. it is a binary file. The information is x y z t dx dy dz E in order.

To run the simulation, use command 
./gPET_test < input_PET.in

When the simulation is finished, three files corresponded to hits, singles and coincidence will be generated. They are stored in binary format. For hits, there are two files hits.dat and hitsID.dat, whose data are t E x y z and particleID, pannelID, moduleID, crystalID and idle one, respectively. For singles and coincidence, the data are particleID, pannelID, moduleID, crystalID, idle one and t E x y z. You can use struct to read them in because they are in interger and float formats, respectively.