This code has been designed to solve the three-phase flow problem in porous media using DG. To run the code, the newest version of deal.ii must be installed with the proper dependencies. If you are using the Sabine UH cluster, then you shouldn't need to download dealii and you can skip to the section on running the code. 

# Installing Dealii on Math Department Computer
When installing anything on a department computer, make sure you install it inside your STORAGE directory and use one of the clusters. To access your storage directory, run 
```
cd /storage/username
```
and it will allow you to make a directory where you can install the library.
 To get into one of the clusters, run 
```
ssh computeXX
```
 on your department computer where XX is the compute number. Normally you can stick to 12,13 or 14. It shouldn't matter which one you use as you can install on one node and use another to run it. 

 DO NOT install inside your scratch directory as it gets wiped periodically and is compute dependent. Try not to install on home as space is limited. If you need more space, contact IT.

## Installing PETSc
PETSc is an external library that handles a lot of the linear algebra used in the code. You first need to download this library and then pass it as a flag when building dealii.
 Your best bet is to pull the repo from https://gitlab.com/petsc/petsc

and then inside the petsc directory run 
```

./configure --with-shared-libraries=1 --download-hypre=1 --download-mumps=1 --download-scalapack=1 --download-metis=1 --download-parmetis=1 --with-debugging=0 --with-x=0
```
Then compile and install and run the proper tests to verify everything is working ok.
```
make all check
```
# Building Dealii
See the dealii website for installation instructions.
https://www.dealii.org/9.6.0/readme.html

The only thing to keep in mind is that you need to include PETSc as flag with the proper link to its' location. You should run the follow command in the build directory inside the dealii directory you pulled from the github which can be found here.
https://github.com/dealii/dealii
```

cmake -DDEAL_II_WITH_PETSC=ON -DCMAKE_INSTALL_PREFIX=/path/to/where/you/want/dealii/installed -DPETSC_DIR=/path/to/where/PETSc/is/installed -DPETSC_ARCH=/path/to/where/petsc/arch/is/installed -DDEAL_II_WITH_MPI=ON -DDEAL_II_WITH_HDF5=OFF ..
```
# Installing Dealii
You then proceed to install dealii. N would be the number of processors you want to utilize to install. Make sure you talk to your Advisor or systems adminstrator on the proper number to use as the program does take quite a long time to install. You then run tests to make sure everything is working properly.

 WARNING: Just because you are able to successfully build and install everything doesn't mean you will pass all the tests. Only proceed using the library if all tests have been passed.

```
make --jobs=N install

make test
```


