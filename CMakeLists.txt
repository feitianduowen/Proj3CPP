cmake_minimum_required(VERSION 3.10)
project(CS209A_Proj3_Matmul C)

# 设置 C 标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED TRUE)

# 全局编译优化选项 (包含 AVX 和 AVX2 指令集支持)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -mavx -mavx2")

# 引入 OpenMP 支持
find_package(OpenMP REQUIRED)

# =========================================================
# 重点：配置 OpenBLAS 库位置
# 由于您的库文件位于项目根目录的 OpenBLAS 文件夹中，这里使用 find_library 精准定位
# 这样能够避免 MinGW 的 ld.exe 报 cannot find -lopenblas 的问题
# =========================================================
find_library(OPENBLAS_LIB 
    NAMES openblas 
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}/OpenBLAS" "${CMAKE_CURRENT_SOURCE_DIR}/lib"
    NO_DEFAULT_PATH
)
if(NOT OPENBLAS_LIB)
    # 如果当前文件夹下没找到，回退到到系统环境变量寻找
    find_library(OPENBLAS_LIB NAMES openblas)
endif()

# 包含 OpenBLAS 头文件 (以免部分机器编译报错)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/OpenBLAS)

# =========================================================
# 将所需的 libopenblas.dll 拷贝到构建输出目录，解决 0xC0000135 找不到 DLL 报错
# =========================================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/OpenBLAS/libopenblas.dll")
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/OpenBLAS/libopenblas.dll" DESTINATION "${CMAKE_BINARY_DIR}")
endif()

# ---------------------------------------------------------
# 1. 编译 test.c 和 main.c 组成的主测试程序
# ---------------------------------------------------------
add_executable(test_matmul src/test.c src/main.c)
# 链接 OpenMP 和确切的 OpenBLAS 库文件
target_link_libraries(test_matmul PRIVATE OpenMP::OpenMP_C ${OPENBLAS_LIB})

# ---------------------------------------------------------
# 2. 编译 temp.c 和 main.c 组成的 temp 测试程序
# ---------------------------------------------------------
add_executable(temp_matmul src/temp.c src/main.c)
target_link_libraries(temp_matmul PRIVATE OpenMP::OpenMP_C ${OPENBLAS_LIB})

# ---------------------------------------------------------
# 3. 编译 ooc.c 和 main.c 组成的核外计算验证程序
# ---------------------------------------------------------
add_executable(ooc_matmul src/ooc.c src/main.c)
target_link_libraries(ooc_matmul PRIVATE OpenMP::OpenMP_C ${OPENBLAS_LIB})
