set(CMAKE_CXX_COMPILER "/project/dsi/apps/easybuild/software/GCCcore/12.3.0/bin/g++")
set(CMAKE_CXX_COMPILER_ARG1 "")
set(CMAKE_CXX_COMPILER_ID "GNU")
set(CMAKE_CXX_COMPILER_VERSION "12.3.0")
set(CMAKE_CXX_COMPILER_VERSION_INTERNAL "")
set(CMAKE_CXX_COMPILER_WRAPPER "")
set(CMAKE_CXX_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_CXX_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CXX_STANDARD_LATEST "23")
set(CMAKE_CXX_COMPILE_FEATURES "cxx_std_98;cxx_template_template_parameters;cxx_std_11;cxx_alias_templates;cxx_alignas;cxx_alignof;cxx_attributes;cxx_auto_type;cxx_constexpr;cxx_decltype;cxx_decltype_incomplete_return_types;cxx_default_function_template_args;cxx_defaulted_functions;cxx_defaulted_move_initializers;cxx_delegating_constructors;cxx_deleted_functions;cxx_enum_forward_declarations;cxx_explicit_conversions;cxx_extended_friend_declarations;cxx_extern_templates;cxx_final;cxx_func_identifier;cxx_generalized_initializers;cxx_inheriting_constructors;cxx_inline_namespaces;cxx_lambdas;cxx_local_type_template_args;cxx_long_long_type;cxx_noexcept;cxx_nonstatic_member_init;cxx_nullptr;cxx_override;cxx_range_for;cxx_raw_string_literals;cxx_reference_qualified_functions;cxx_right_angle_brackets;cxx_rvalue_references;cxx_sizeof_member;cxx_static_assert;cxx_strong_enums;cxx_thread_local;cxx_trailing_return_types;cxx_unicode_literals;cxx_uniform_initialization;cxx_unrestricted_unions;cxx_user_literals;cxx_variadic_macros;cxx_variadic_templates;cxx_std_14;cxx_aggregate_default_initializers;cxx_attribute_deprecated;cxx_binary_literals;cxx_contextual_conversions;cxx_decltype_auto;cxx_digit_separators;cxx_generic_lambdas;cxx_lambda_init_captures;cxx_relaxed_constexpr;cxx_return_type_deduction;cxx_variable_templates;cxx_std_17;cxx_std_20;cxx_std_23")
set(CMAKE_CXX98_COMPILE_FEATURES "cxx_std_98;cxx_template_template_parameters")
set(CMAKE_CXX11_COMPILE_FEATURES "cxx_std_11;cxx_alias_templates;cxx_alignas;cxx_alignof;cxx_attributes;cxx_auto_type;cxx_constexpr;cxx_decltype;cxx_decltype_incomplete_return_types;cxx_default_function_template_args;cxx_defaulted_functions;cxx_defaulted_move_initializers;cxx_delegating_constructors;cxx_deleted_functions;cxx_enum_forward_declarations;cxx_explicit_conversions;cxx_extended_friend_declarations;cxx_extern_templates;cxx_final;cxx_func_identifier;cxx_generalized_initializers;cxx_inheriting_constructors;cxx_inline_namespaces;cxx_lambdas;cxx_local_type_template_args;cxx_long_long_type;cxx_noexcept;cxx_nonstatic_member_init;cxx_nullptr;cxx_override;cxx_range_for;cxx_raw_string_literals;cxx_reference_qualified_functions;cxx_right_angle_brackets;cxx_rvalue_references;cxx_sizeof_member;cxx_static_assert;cxx_strong_enums;cxx_thread_local;cxx_trailing_return_types;cxx_unicode_literals;cxx_uniform_initialization;cxx_unrestricted_unions;cxx_user_literals;cxx_variadic_macros;cxx_variadic_templates")
set(CMAKE_CXX14_COMPILE_FEATURES "cxx_std_14;cxx_aggregate_default_initializers;cxx_attribute_deprecated;cxx_binary_literals;cxx_contextual_conversions;cxx_decltype_auto;cxx_digit_separators;cxx_generic_lambdas;cxx_lambda_init_captures;cxx_relaxed_constexpr;cxx_return_type_deduction;cxx_variable_templates")
set(CMAKE_CXX17_COMPILE_FEATURES "cxx_std_17")
set(CMAKE_CXX20_COMPILE_FEATURES "cxx_std_20")
set(CMAKE_CXX23_COMPILE_FEATURES "cxx_std_23")
set(CMAKE_CXX26_COMPILE_FEATURES "")

set(CMAKE_CXX_PLATFORM_ID "Linux")
set(CMAKE_CXX_SIMULATE_ID "")
set(CMAKE_CXX_COMPILER_FRONTEND_VARIANT "GNU")
set(CMAKE_CXX_SIMULATE_VERSION "")




set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_CXX_COMPILER_AR "/project/dsi/apps/easybuild/software/GCCcore/12.3.0/bin/gcc-ar")
set(CMAKE_RANLIB "/usr/bin/ranlib")
set(CMAKE_CXX_COMPILER_RANLIB "/project/dsi/apps/easybuild/software/GCCcore/12.3.0/bin/gcc-ranlib")
set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_LINKER_LINK "")
set(CMAKE_LINKER_LLD "")
set(CMAKE_CXX_COMPILER_LINKER "/project/dsi/apps/easybuild/software/binutils/2.40-GCCcore-12.3.0/bin/ld")
set(CMAKE_CXX_COMPILER_LINKER_ID "GNU")
set(CMAKE_CXX_COMPILER_LINKER_VERSION 2.40)
set(CMAKE_CXX_COMPILER_LINKER_FRONTEND_VARIANT GNU)
set(CMAKE_MT "")
set(CMAKE_TAPI "CMAKE_TAPI-NOTFOUND")
set(CMAKE_COMPILER_IS_GNUCXX 1)
set(CMAKE_CXX_COMPILER_LOADED 1)
set(CMAKE_CXX_COMPILER_WORKS TRUE)
set(CMAKE_CXX_ABI_COMPILED TRUE)

set(CMAKE_CXX_COMPILER_ENV_VAR "CXX")

set(CMAKE_CXX_COMPILER_ID_RUN 1)
set(CMAKE_CXX_SOURCE_FILE_EXTENSIONS C;M;c++;cc;cpp;cxx;m;mm;mpp;CPP;ixx;cppm;ccm;cxxm;c++m)
set(CMAKE_CXX_IGNORE_EXTENSIONS inl;h;hpp;HPP;H;o;O;obj;OBJ;def;DEF;rc;RC)

foreach (lang IN ITEMS C OBJC OBJCXX)
  if (CMAKE_${lang}_COMPILER_ID_RUN)
    foreach(extension IN LISTS CMAKE_${lang}_SOURCE_FILE_EXTENSIONS)
      list(REMOVE_ITEM CMAKE_CXX_SOURCE_FILE_EXTENSIONS ${extension})
    endforeach()
  endif()
endforeach()

set(CMAKE_CXX_LINKER_PREFERENCE 30)
set(CMAKE_CXX_LINKER_PREFERENCE_PROPAGATES 1)
set(CMAKE_CXX_LINKER_DEPFILE_SUPPORTED FALSE)

# Save compiler ABI information.
set(CMAKE_CXX_SIZEOF_DATA_PTR "8")
set(CMAKE_CXX_COMPILER_ABI "ELF")
set(CMAKE_CXX_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CXX_LIBRARY_ARCHITECTURE "")

if(CMAKE_CXX_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CXX_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CXX_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CXX_COMPILER_ABI}")
endif()

if(CMAKE_CXX_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX "")
if(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX)
  set(CMAKE_CL_SHOWINCLUDES_PREFIX "${CMAKE_CXX_CL_SHOWINCLUDES_PREFIX}")
endif()





set(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "/project/dsi/apps/easybuild/software/deal.II/9.6.0-foss-2023a/include/deal.II/bundled;/project/dsi/apps/easybuild/software/deal.II/9.6.0-foss-2023a/include;/project/dsi/apps/easybuild/software/PETSc/3.20.3-foss-2023a/include;/project/dsi/apps/easybuild/software/SuperLU_DIST/8.1.2-foss-2023a/include;/project/dsi/apps/easybuild/software/ParMETIS/4.0.3-gompi-2023a/include;/project/dsi/apps/easybuild/software/Hypre/2.29.0-foss-2023a/include;/project/dsi/apps/easybuild/software/SuiteSparse/7.1.0-foss-2023a/include;/project/dsi/apps/easybuild/software/MPFR/4.2.0-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/GMP/6.2.1-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/MUMPS/5.6.1-foss-2023a-metis/include;/project/dsi/apps/easybuild/software/SCOTCH/7.0.3-gompi-2023a/include;/project/dsi/apps/easybuild/software/SciPy-bundle/2023.07-gfbf-2023a/lib/python3.11/site-packages/numpy/core/include;/project/dsi/apps/easybuild/software/pybind11/2.11.1-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/Python/3.11.3-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/libffi/3.4.4-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/SQLite/3.42.0-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/Tcl/8.6.13-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/libreadline/8.2-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/p4est/2.8.6-foss-2023a/include;/project/dsi/apps/easybuild/software/METIS/5.1.0-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/HDF5/1.14.0-gompi-2023a/include;/project/dsi/apps/easybuild/software/Szip/2.1.1-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/GSL/2.7-GCC-12.3.0/include;/project/dsi/apps/easybuild/software/Boost/1.82.0-GCC-12.3.0/include;/project/dsi/apps/easybuild/software/ICU/73.2-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/FFTW.MPI/3.3.10-gompi-2023a/include;/project/dsi/apps/easybuild/software/FFTW/3.3.10-GCC-12.3.0/include;/project/dsi/apps/easybuild/software/FlexiBLAS/3.3.1-GCC-12.3.0/include;/project/dsi/apps/easybuild/software/libarchive/3.6.2-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/cURL/8.0.1-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/OpenSSL/1.1/include;/project/dsi/apps/easybuild/software/bzip2/1.0.8/include;/project/dsi/apps/easybuild/software/ncurses/6.4/include;/project/dsi/apps/easybuild/software/OpenBLAS/0.3.23-GCC-12.3.0/include;/project/dsi/apps/easybuild/software/OpenMPI/4.1.5-GCC-12.3.0/include;/project/dsi/apps/easybuild/software/UCC/1.2.0-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/PMIx/4.2.4-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/libfabric/1.18.0-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/UCX/1.14.1-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/zstd/1.5.5-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/lz4/1.9.4-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/libevent/2.1.12/include;/project/dsi/apps/easybuild/software/hwloc/2.9.1-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/libpciaccess/0.17-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/libxml2/2.11.4-GCCcore-12.3.0/include/libxml2;/project/dsi/apps/easybuild/software/libxml2/2.11.4-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/XZ/5.4.2-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/numactl/2.0.16-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/binutils/2.40-GCCcore-12.3.0/include;/project/dsi/apps/easybuild/software/zlib/1.2.13/include;/project/dsi/apps/easybuild/software/GCCcore/12.3.0/include/c++/12.3.0;/project/dsi/apps/easybuild/software/GCCcore/12.3.0/include/c++/12.3.0/x86_64-pc-linux-gnu;/project/dsi/apps/easybuild/software/GCCcore/12.3.0/include/c++/12.3.0/backward;/project/dsi/apps/easybuild/software/GCCcore/12.3.0/lib/gcc/x86_64-pc-linux-gnu/12.3.0/include;/project/dsi/apps/easybuild/software/GCCcore/12.3.0/include;/project/dsi/apps/easybuild/software/GCCcore/12.3.0/lib/gcc/x86_64-pc-linux-gnu/12.3.0/include-fixed;/usr/include")
set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "/project/dsi/apps/easybuild/software/deal.II/9.6.0-foss-2023a/lib64;/project/dsi/apps/easybuild/software/PETSc/3.20.3-foss-2023a/lib64;/project/dsi/apps/easybuild/software/mpi4py/3.1.4-gompi-2023a/lib64;/project/dsi/apps/easybuild/software/SuperLU_DIST/8.1.2-foss-2023a/lib64;/project/dsi/apps/easybuild/software/ParMETIS/4.0.3-gompi-2023a/lib64;/project/dsi/apps/easybuild/software/Hypre/2.29.0-foss-2023a/lib64;/project/dsi/apps/easybuild/software/SuiteSparse/7.1.0-foss-2023a/lib64;/project/dsi/apps/easybuild/software/MPFR/4.2.0-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/GMP/6.2.1-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/MUMPS/5.6.1-foss-2023a-metis/lib64;/project/dsi/apps/easybuild/software/SCOTCH/7.0.3-gompi-2023a/lib64;/project/dsi/apps/easybuild/software/SciPy-bundle/2023.07-gfbf-2023a/lib64;/project/dsi/apps/easybuild/software/Python-bundle-PyPI/2023.06-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/virtualenv/20.23.1-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/cffi/1.15.1-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/Python/3.11.3-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/libffi/3.4.4-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/SQLite/3.42.0-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/Tcl/8.6.13-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/libreadline/8.2-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/p4est/2.8.6-foss-2023a/lib64;/project/dsi/apps/easybuild/software/METIS/5.1.0-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/HDF5/1.14.0-gompi-2023a/lib64;/project/dsi/apps/easybuild/software/Szip/2.1.1-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/GSL/2.7-GCC-12.3.0/lib64;/project/dsi/apps/easybuild/software/Boost/1.82.0-GCC-12.3.0/lib64;/project/dsi/apps/easybuild/software/ICU/73.2-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/ScaLAPACK/2.2.0-gompi-2023a-fb/lib64;/project/dsi/apps/easybuild/software/FFTW.MPI/3.3.10-gompi-2023a/lib64;/project/dsi/apps/easybuild/software/FFTW/3.3.10-GCC-12.3.0/lib64;/project/dsi/apps/easybuild/software/FlexiBLAS/3.3.1-GCC-12.3.0/lib64;/project/dsi/apps/easybuild/software/libarchive/3.6.2-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/cURL/8.0.1-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/OpenSSL/1.1/lib64;/project/dsi/apps/easybuild/software/bzip2/1.0.8/lib64;/project/dsi/apps/easybuild/software/ncurses/6.4/lib64;/project/dsi/apps/easybuild/software/OpenBLAS/0.3.23-GCC-12.3.0/lib64;/project/dsi/apps/easybuild/software/OpenMPI/4.1.5-GCC-12.3.0/lib64;/project/dsi/apps/easybuild/software/UCC/1.2.0-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/PMIx/4.2.4-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/libfabric/1.18.0-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/UCX/1.14.1-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/zstd/1.5.5-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/lz4/1.9.4-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/libevent/2.1.12/lib64;/project/dsi/apps/easybuild/software/hwloc/2.9.1-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/libpciaccess/0.17-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/libxml2/2.11.4-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/XZ/5.4.2-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/numactl/2.0.16-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/binutils/2.40-GCCcore-12.3.0/lib64;/project/dsi/apps/easybuild/software/zlib/1.2.13/lib64;/project/dsi/apps/easybuild/software/GCCcore/12.3.0/lib/gcc/x86_64-pc-linux-gnu/12.3.0;/project/dsi/apps/easybuild/software/GCCcore/12.3.0/lib64;/lib64;/usr/lib64;/project/dsi/apps/easybuild/software/deal.II/9.6.0-foss-2023a/lib;/project/dsi/apps/easybuild/software/PETSc/3.20.3-foss-2023a/lib;/project/dsi/apps/easybuild/software/mpi4py/3.1.4-gompi-2023a/lib;/project/dsi/apps/easybuild/software/SuperLU_DIST/8.1.2-foss-2023a/lib;/project/dsi/apps/easybuild/software/ParMETIS/4.0.3-gompi-2023a/lib;/project/dsi/apps/easybuild/software/Hypre/2.29.0-foss-2023a/lib;/project/dsi/apps/easybuild/software/SuiteSparse/7.1.0-foss-2023a/lib;/project/dsi/apps/easybuild/software/MPFR/4.2.0-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/GMP/6.2.1-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/MUMPS/5.6.1-foss-2023a-metis/lib;/project/dsi/apps/easybuild/software/SCOTCH/7.0.3-gompi-2023a/lib;/project/dsi/apps/easybuild/software/SciPy-bundle/2023.07-gfbf-2023a/lib/python3.11/site-packages/numpy/core/lib;/project/dsi/apps/easybuild/software/SciPy-bundle/2023.07-gfbf-2023a/lib;/project/dsi/apps/easybuild/software/Python-bundle-PyPI/2023.06-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/virtualenv/20.23.1-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/cffi/1.15.1-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/Python/3.11.3-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/libffi/3.4.4-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/SQLite/3.42.0-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/Tcl/8.6.13-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/libreadline/8.2-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/p4est/2.8.6-foss-2023a/lib;/project/dsi/apps/easybuild/software/METIS/5.1.0-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/HDF5/1.14.0-gompi-2023a/lib;/project/dsi/apps/easybuild/software/Szip/2.1.1-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/GSL/2.7-GCC-12.3.0/lib;/project/dsi/apps/easybuild/software/Boost/1.82.0-GCC-12.3.0/lib;/project/dsi/apps/easybuild/software/ICU/73.2-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/ScaLAPACK/2.2.0-gompi-2023a-fb/lib;/project/dsi/apps/easybuild/software/FFTW.MPI/3.3.10-gompi-2023a/lib;/project/dsi/apps/easybuild/software/FFTW/3.3.10-GCC-12.3.0/lib;/project/dsi/apps/easybuild/software/FlexiBLAS/3.3.1-GCC-12.3.0/lib;/project/dsi/apps/easybuild/software/libarchive/3.6.2-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/cURL/8.0.1-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/OpenSSL/1.1/lib;/project/dsi/apps/easybuild/software/bzip2/1.0.8/lib;/project/dsi/apps/easybuild/software/ncurses/6.4/lib;/project/dsi/apps/easybuild/software/OpenBLAS/0.3.23-GCC-12.3.0/lib;/project/dsi/apps/easybuild/software/OpenMPI/4.1.5-GCC-12.3.0/lib;/project/dsi/apps/easybuild/software/UCC/1.2.0-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/PMIx/4.2.4-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/libfabric/1.18.0-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/UCX/1.14.1-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/zstd/1.5.5-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/lz4/1.9.4-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/libevent/2.1.12/lib;/project/dsi/apps/easybuild/software/hwloc/2.9.1-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/libpciaccess/0.17-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/libxml2/2.11.4-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/XZ/5.4.2-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/numactl/2.0.16-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/binutils/2.40-GCCcore-12.3.0/lib;/project/dsi/apps/easybuild/software/zlib/1.2.13/lib;/project/dsi/apps/easybuild/software/GCCcore/12.3.0/lib")
set(CMAKE_CXX_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
set(CMAKE_CXX_COMPILER_CLANG_RESOURCE_DIR "")

set(CMAKE_CXX_COMPILER_IMPORT_STD "")
### Imported target for C++23 standard library
set(CMAKE_CXX23_COMPILER_IMPORT_STD_NOT_FOUND_MESSAGE "Unsupported generator: Unix Makefiles")



