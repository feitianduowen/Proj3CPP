gcc test.c main.c -O3 -mavx -mavx2 -o test -L. -lopenblas



gcc test.c main.c -O3 -mavx2 -fopenmp -o test -L. -lopenblas

gcc temp.c main.c -O3 -mavx2 -fopenmp -o temp -L. -lopenblas

gcc ooc.c main.c -O3 -mavx2 -fopenmp -o ooc -L. -lopenblas

.\ooc