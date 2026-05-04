gcc test.c main.c -O3 -mavx -mavx2 -o ../out/test -L../OpenBLAS -lopenblas



gcc test.c main.c -O3 -mavx2 -fopenmp -o ../out/test -L../OpenBLAS -lopenblas

gcc temp.c main.c -O3 -mavx2 -fopenmp -o ../out/temp -L../OpenBLAS -lopenblas

gcc ooc.c main.c -O3 -mavx2 -fopenmp -o ../out/ooc -L../OpenBLAS -lopenblas

.\ooc