cd ./gPETphantom
./gbrachy_test < input_PET.in 
cp ./phaseSpace.dat ../gPETdetector/patientcase/PETConfiguration/
cd ../gPETdetector
./gPET_test < input_PET.in
