# Three Phase Flow Code


This code has been designed to solve the three-phase flow problem in porous media using DG. To run the code, the newest version of deal.
ii must be installed with the proper dependencies. If you are using the Sabine UH cluster, then you shouldn't need to download dealii and you can skip to the section on running the code. 

# Installing Deal.ii on UH Math Department Computer
Installing deal.ii can be kind of a pain, especially if you are not familiar with using the command line. Make sure you're comfortable with navigating the command line first before installing. There are a lot of helpful videos that can be found here https://www.math.colostate.edu/~bangerth/videos.html
that serve to better understand installation of large software libraries and FEM theory.

NOTE: It should be noted here that you will not be able to use any IDE's or any microsoft text editors. I have tried using CLION and it works for the most part, but you're best bet is to either work from the terminal using a text editor like vim or nano, or KWrite which is installed already on the department computer.
# Loging Into Your Department Computer
Remote access is available to you if you don't want to use your provided department computer. If You have a Windows computer I recommend VSCode to ssh into the department computer. You can use things like PuTTY or XCode, but VSCode is the best modern solution. See https://code.visualstudio.com/docs/remote/ssh for more details. If you have a mac, you can just use the terminal and run 
```
ssh your_username@ssh.math.uh.edu
```
You will then be prompted to enter your password and then you will be able to have access.

NOTE: You will only be on the login node. If you want to run anything you will need to access a compute node which I will briefly discuss below.

NOTE: You can run the same command in the Ubuntu terminal if you're using a WSL for windows. 

# Installing Software On Your Department Computer

Do note that you will be limited to the software you are allowed to install on your computer. DO NOT USE THE SUDO COMMAND UNDER ANY CIRCUMSTANCE. The department is super strict about this and it could result in loss of computer privledges.

When installing anything on a department computer, it's best to install it inside your STORAGE directory and use one of the clusters. To access your storage directory, run 
```
cd /storage/username
```
and it will allow you to make a directory where you can install the library.
 To login to the UH math department cluster run
```
ssh computeXX
```
 on your department computer where X is the compute number. Normally you can stick to 12,13 or 14. It shouldn't matter which one you use as you can install on one node and use another to run it. 

 DO NOT install inside your scratch directory as it gets wiped periodically and is compute dependent. Try not to install on home as space is limited. If you need more space, contact IT at help.math.uh.edu
## Installing PETSc
PETSc is an external library that handles a lot of the linear algebra used in the code. You first need to download this library and then pass it as a flag when building dealii.
 Your best bet is to pull the repo from https://gitlab.com/petsc/petsc

and then inside the petsc directory run 
```

./configure --with-shared-libraries=1 --download-hypre=1 --download-mumps=1 --download-scalapack=1 --download-metis=1 --download-parmetis=1 --with-debugging=0 --with-x=0
```
You should keep this in a directory separate from where you install dealii. Then compile and install and run the proper tests to verify everything is working ok.
```
make all check
```
## Building Dealii
See the dealii website for detailed installation instructions.
https://www.dealii.org/9.6.0/readme.html

The only thing to keep in mind is that you need to include PETSc as flag with the proper link to its' location. You should run the follow command in the build directory inside the dealii directory you pulled from the github which can be found here.
https://github.com/dealii/dealii
```

cmake -DDEAL_II_WITH_PETSC=ON -DCMAKE_INSTALL_PREFIX=/path/to/where/you/want/dealii/installed -DPETSC_DIR=/path/to/where/PETSc/is/installed -DPETSC_ARCH=/path/to/where/petsc/arch/is/installed -DDEAL_II_WITH_MPI=ON -DDEAL_II_WITH_HDF5=OFF ..
```
## Installing Deal.ii
You then proceed to install deal.ii.
run 

```
make --jobs=N install

make test
``` 
where N is the number of processors you want to utilize to install. Make sure you talk to your dvisor or systems adminstrator on the proper number to use as the program does take quite a long time to install. You then run tests to make sure everything is working properly.

WARNING: Just because you are able to successfully build and install everything doesn't mean you will pass all the tests. Only proceed using the library if all tests have been passed.

# Running The Code

Running the code is pretty straight forward, but you need to be careful on what problem you are solving. The code is meant to be robust and at this point it handles 3 different problems. The main distinguister are the two files 
```
Auxiliary_Functions_PrimaryVars.cc
paramters.prm
```
that give the necessary problem parameters. You will find those in 
```
\MANUFACTURED_TEST
\QUARTER_5_SPOT
\VISC_FING
```
By default, the code should have the manufactured test case in the main directory. Just copy from the desired directory and put them in the main directory. Just make sure that when you are done that you save your changes. 

Next,  run 
```
cmake .
```
to generate the necessary build files. Then
``` 
make 
```
to compile. you can in addition run 
```
make release
```
but it's not strictly necessary. Once everything is compiled, you can run the code via
```
.\ThreePhase
```
or 
```
mpirun -np N .\ThreePhase
```
if you want to run in parallel. N is the number of processors you give it. 