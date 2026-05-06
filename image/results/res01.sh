D:\comp sci\CS209A cpp\lab\project\proj_3\src>.\test
Testing N = 16, cir = 10000
  OpenBLAS time: 428.0000 Ns
  Improved time: 377.0000 Ns
  Plain time:    1872.0000 Ns
  Check Improved: PASSED
  Speedup (Improved over Plain): 4.97x
  Speedup (OpenBLAS over Plain): 4.37x

Testing N = 128, cir = 1250
  OpenBLAS time: 117318.0000 Ns
  Improved time: 129552.0000 Ns
  Plain time:    1060041.0000 Ns
  Check Improved: PASSED
  Speedup (Improved over Plain): 8.18x
  Speedup (OpenBLAS over Plain): 9.04x

Testing N = 400, cir = 400
  OpenBLAS time: 2203841.0000 Ns
  OpenBLAS time: 2203841.0000 Ns
  Improved time: 3964302.0000 Ns
  Plain time:    28040589.0000 Ns
  Check Improved: PASSED
  Speedup (Improved over Plain): 7.07x
  Speedup (OpenBLAS over Plain): 12.72x

Testing N = 800, cir = 200
  OpenBLAS time: 6012978.0000 Ns
  Improved time: 67478681.0000 Ns
  Plain time:    297509660.0000 Ns
  Check Improved: PASSED
  Speedup (Improved over Plain): 4.41x
  Speedup (OpenBLAS over Plain): 49.48x

Testing N = 1024, cir = 10
  OpenBLAS time: 43779260.0000 Ns
  Improved time: 346362070.0000 Ns
  Plain time:    3110647670.0000 Ns
  Check Improved: PASSED
  Speedup (Improved over Plain): 8.98x
  Speedup (OpenBLAS over Plain): 71.05x

Testing N = 8192, cir = 10


+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

N = 1024 , blocksize = 256 , 80957600 ns! #4

N = 1024 , blocksize = 512 , 28814600 ns! #2

N = 2048 , blocksize = 256 , 645652700 ns! #8

N = 2048 , blocksize = 512 , 228383200 ns! #4

N = 4096 , blocksize = 256 , 5306003300 ns! #16

N = 4096 , blocksize = 512 , 1780625400 ns! #8

N = 4096 , blocksize = 1024 , 525225000 ns! #4

N = 8192 , blocksize = 256 , 44703255800 ns! #32

N = 8192 , blocksize = 512 , 14737828400 ns! #16

N = 8192 , blocksize = 1024 , 4014717600 ns! #8

N = 8192 , blocksize = 2048 , 1935803600 ns! #4

N = 16384 , blocksize = 256 , 375222695400 ns! #64

N = 16384 , blocksize = 512 , 126022599000 ns! #32

N = 16384 , blocksize = 1024 , 41832027300 ns! #16

N = 16384 , blocksize = 2048 , 30539487177 ns! #8

N = 16384 , blocksize = 4096 , 10800783900 ns! #4

N = 65536 , blocksize = 4096 , 812786511100 ns! #16

N = 65536 , blocksize = 8192 , 637611920600 ns! #8

N = 65536 , blocksize = 16384 , 622136449800 ns! #4
++++++++++++++++++++++++++++

"D:\comp sci\CS209A cpp\project\proj_3\cmake-build-debug\test_matmul.exe"
 N = 16, outer 10, plain 4600, improved 400, openblas 1100
 N = 128, outer 1000, plain 775800, improved 48200, openblas 40700
 N = 256, outer 1000, plain 15154900, improved 739500, openblas 714500
 N = 512, outer 100, plain 60927600, improved 756300, openblas 443200
 N = 1024, outer 10, plain 2300996000, improved 3856500, openblas 3138200
 N = 2048, outer 10, plain 33709007300, improved 24285100, openblas 15552200
 N = 8192, outer 5, plain 6568479443800, improved 3556561500, openblas 949965400

==> Results successfully saved to ../out/result.csv

进程已结束，退出代码为 0

+++++++++++++++++++++++++++++++++++++

"D:\comp sci\CS209A cpp\lab\project\proj_3\cmake-build-debug\temp_matmul.exe"
plain: 1000 ns, tp: 100 ns
Check for N = 16 PASSED
plain: 31100 ns, tp: 7800 ns
Check for N = 50 PASSED
plain: 680100 ns, tp: 69400 ns
Check for N = 128 PASSED
plain: 6320600 ns, tp: 292900 ns
Check for N = 256 PASSED
plain: 20970700 ns, tp: 810200 ns
Check for N = 400 PASSED
plain: 233436200 ns, tp: 2770900 ns
Check for N = 800 PASSED
plain: 2090204700 ns, tp: 17185100 ns
Check for N = 1600 PASSED
plain: 11915425600 ns, tp: 33106500 ns
Check for N = 2000 PASSED

+++++++++++++++++++++++++++++++++++

"D:\comp sci\CS209A cpp\project\proj_3\cmake-build-debug\temp_matmul.exe"
#ALIGNED
size,time
16,100
64,4900
128,39700
256,148700
512,561100
1024,3433700
2048,24101300
4096,282670900
8192,3660994100
进程已结束，退出代码为 0

"D:\comp sci\CS209A cpp\project\proj_3\cmake-build-debug\temp_matmul.exe"
#UNALIGNED
size,time
16,100
64,4900
128,40500
256,155900
512,464500
1024,3584900
2048,23975700
4096,268111300
8192,3677244200
进程已结束，退出代码为 0

++++++++++++++++++++++++++++++++++++++++

"D:\comp sci\CS209A cpp\project\proj_3\cmake-build-debug\test_matmul.exe"
improved: 100 ns, aligned: 100 ns
Check for N = 16 PASSED
improved: 5000 ns, aligned: 5000 ns
Check for N = 64 PASSED
improved: 40100 ns, aligned: 39600 ns
Check for N = 128 PASSED
improved: 155900 ns, aligned: 151700 ns
Check for N = 256 PASSED
improved: 557300 ns, aligned: 466600 ns
Check for N = 512 PASSED
improved: 3304600 ns, aligned: 3332100 ns
Check for N = 1024 PASSED
improved: 24525300 ns, aligned: 25036100 ns
Check for N = 2048 PASSED
improved: 277543900 ns, aligned: 279143300 ns
Check for N = 4096 PASSED
improved: 3608518700 ns, aligned: 3553906100 ns
Check for N = 8192 PASSED

进程已结束，退出代码为 0
"D:\comp sci\CS209A cpp\project\proj_3\cmake-build-debug\test_matmul.exe"
aligned: 200 ns, improved: 500 ns
Check for N = 16 PASSED
aligned: 5300 ns, improved: 6900 ns
Check for N = 64 PASSED
aligned: 40400 ns, improved: 46500 ns
Check for N = 128 PASSED
aligned: 171300 ns, improved: 231900 ns
Check for N = 256 PASSED
aligned: 652800 ns, improved: 647800 ns
Check for N = 512 PASSED
aligned: 3456700 ns, improved: 3679000 ns
Check for N = 1024 PASSED
aligned: 25270600 ns, improved: 25666500 ns
Check for N = 2048 PASSED
aligned: 308666500 ns, improved: 294842600 ns
Check for N = 4096 PASSED
aligned: 3762494200 ns, improved: 3787315900 ns
Check for N = 8192 PASSED

进程已结束，退出代码为 0

+++++++++++++++++
ALIGNED
16,100
64,9000
128,41700
256,274300
512,931500
1024,7988600
2048,49024400
4096,534566400
8192,5906959600
UNALIGNED
16,100
64,5700
128,65800
256,323400
512,1234900
1024,8668800
2048,58381200
4096,466038600
8192,6895026900


++++++

"D:\comp sci\CS209A cpp\project\proj_3\cmake-build-debug\test_matmul.exe"
Check for plain -> strassen      N = 1024, max diff = 2.380371e-03, mean diff = 1.771608e+02, cnt6 = 530644, cnt5 = 0, c
nt4= 3959826, cnt3 = 5954729, cnt2 =40561 .
Check for plain -> improved      N = 1024, max diff = 2.304077e-03, mean diff = 1.773924e+02, cnt6 = 534102, cnt5 = 0, c
nt4= 3955868, cnt3 = 5954467, cnt2 =41323 .
Check for plain -> aligned       N = 1024, max diff = 2.258301e-03, mean diff = 1.772305e+02, cnt6 = 534627, cnt5 = 0, c
nt4= 3957304, cnt3 = 5953156, cnt2 =40673 .
Check for plain -> openblas      N = 1024, max diff = 6.408691e-04, mean diff = 9.492830e+01, cnt6 = 835136, cnt5 = 0, c
nt4= 5826991, cnt3 = 3823633, cnt2 =0 .

进程已结束，退出代码为 0

+++++++

"D:\comp sci\CS209A cpp\lab\project\proj_3\cmake-build-debug\test_matmul.exe"
Check for plain -> strassen      N = 1024, max diff = 2.380371e-03, mean diff = 1.689537e-04, cnt7=530644, cnt6 = 0, cnt
5 = 0, cnt41= 1761056, cnt42 = 2198770, cnt31 = 5511739, cnt32 = 442990, cnt2 =40561 .
Check for plain -> improved      N = 1024, max diff = 2.304077e-03, mean diff = 1.691746e-04, cnt7=534102, cnt6 = 0, cnt
5 = 0, cnt41= 1751641, cnt42 = 2204227, cnt31 = 5509812, cnt32 = 444655, cnt2 =41323 .
Check for plain -> aligned       N = 1024, max diff = 2.258301e-03, mean diff = 1.690202e-04, cnt7=534627, cnt6 = 0, cnt
5 = 0, cnt41= 1751536, cnt42 = 2205768, cnt31 = 5509319, cnt32 = 443837, cnt2 =40673 .
Check for plain -> openblas      N = 1024, max diff = 6.408691e-04, mean diff = 9.028908e-05, cnt7=837394, cnt6 = 0, cnt
5 = 0, cnt41= 2717052, cnt42 = 3121532, cnt31 = 3809648, cnt32 = 134, cnt2 =0 .

进程已结束，退出代码为 0

++++++++++++

"D:\comp sci\CS209A cpp\lab\project\proj_3\cmake-build-debug\test_matmul.exe"
Check for plain -> strassen      N = 1024, max = 2.380371e-03, mean = 1.689537e-04, cnt0=530644, 41=0, 42=368393, 43=0,
44=1040109, 45=352554, 46=0, 47=980863, 48=325399, 49=0, 31=892508, 32=2913888, 33=1428734, 34=843180, 35=325937, 36=192
815, 37=104663, 38=78680, 39=39203, >1e-3=68190.
Check for plain -> improved      N = 1024, max = 2.304077e-03, mean = 1.691746e-04, cnt0=534102, 41=0, 42=360216, 43=0,
44=1045801, 45=345624, 46=0, 47=986625, 48=319214, 49=0, 31=898388, 32=2908764, 33=1431361, 34=844464, 35=325223, 36=193
309, 37=104872, 38=78899, 39=39649, >1e-3=69249.
Check for plain -> aligned       N = 1024, max = 2.258301e-03, mean = 1.690202e-04, cnt0=534627, 41=0, 42=359609, 43=0,
44=1048056, 45=343871, 46=0, 47=989554, 48=317019, 49=0, 31=899195, 32=2908536, 33=1432094, 34=845240, 35=323449, 36=193
035, 37=104209, 38=79430, 39=39553, >1e-3=68283.
Check for plain -> openblas      N = 1024, max = 6.408691e-04, mean = 9.028908e-05, cnt0=837394, 41=0, 42=572141, 43=0,
44=1615044, 45=529867, 46=0, 47=1454029, 48=453532, 49=0, 31=1213971, 32=3018388, 33=684743, 34=102963, 35=3554, 36=132,
 37=2, 38=0, 39=0, >1e-3=0.

进程已结束，退出代码为 0
