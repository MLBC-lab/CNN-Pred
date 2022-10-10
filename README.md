
******************************** To use this implementation, you will need these libraries in python: *******************************  
1) scipy.io
2) inspect
3) dis
4) numpy 
5) keras
6) os
7) sklearn


*************************************************** Guideline to code execution: ***************************************************

The file under the name "Data_DSB_SSB.mat" is the raw dataset. 

The first code to be executed is the file under the name "One_Convert_Data_from_Matlab_to_python" and the input of this code is the mentioned dataset.

This code converts the dataset in MATLAB format into files needed to run in Python.

The output of this code is the PSSM matrices, which are stored in the PSSM folder!

After obtaining the PSSM matrix, the second code called "Two_Extract_Mono_and_Bigram" must be executed to extract Monogram and Bigram combinations.

Finally, after obtaining Monogram and Bigram combinations, the neural network should be trained.

The network training and evaluation codes are located in the "Three_Classification_Model" file.
