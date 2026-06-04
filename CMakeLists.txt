cmake_minimum_required(VERSION 3.10)
project(QuantANNs LANGUAGES C CXX)

message(STATUS "CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")

set(CMAKE_CXX_STANDARD 17)

# 1. 自动检测处理器
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
    set(ARCH "arm")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i[3-6]86|x86_64|AMD64")
    set(ARCH "x86")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# 2. 根据架构加选项
if(ARCH STREQUAL "arm")
    add_compile_options(-Ofast -fopenmp-simd -funroll-loops -std=c++17 -march=native -falign-jumps=64)
    add_definitions(-DARM_BUILD)
else()
    add_compile_options(-Ofast -fopenmp -fopenmp-simd -funroll-loops -std=c++17 -march=native -falign-jumps=64 -mcx16)
    # add_compile_options(-fopenmp -fopenmp-simd -std=c++17)
    add_definitions(-DX86_BUILD)
endif()

# 递归查找所有 .cpp 文件（给 gcc/clang）
file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithms/utils/*.cpp",
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithms/format/*.cpp"
)

find_package(ZLIB REQUIRED)

# 格式转换
add_executable(FmtConv ${CMAKE_CURRENT_SOURCE_DIR}/data_tools/ANNs/data_proc/FmtConv.cpp ${CPP_SOURCES})
target_compile_definitions(FmtConv PRIVATE STATS NDEBUG)
target_compile_options(FmtConv PRIVATE -fopenmp)
target_include_directories(FmtConv
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/algorithms
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(FmtConv PRIVATE -lgomp ZLIB::ZLIB)

# compute groundtruth
add_executable(CompGT ${CMAKE_CURRENT_SOURCE_DIR}/data_tools/ANNs/data_proc/CompGT.cpp ${CPP_SOURCES})
target_compile_definitions(CompGT PRIVATE STATS NDEBUG)
target_compile_options(CompGT PRIVATE -fopenmp)
target_include_directories(CompGT
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/algorithms
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(CompGT PRIVATE -lgomp ZLIB::ZLIB)

# pca compress
add_executable(PCACompress ${CMAKE_CURRENT_SOURCE_DIR}/data_tools/ANNs/data_proc/PCACompress.cpp ${CPP_SOURCES})
target_compile_definitions(PCACompress PRIVATE STATS NDEBUG)
target_compile_options(PCACompress PRIVATE -fopenmp)
target_include_directories(PCACompress
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/algorithms
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(PCACompress PRIVATE -lgomp ZLIB::ZLIB)


# pca test
# add_executable(PCATest ${CMAKE_CURRENT_SOURCE_DIR}/data_tools/ANNs/data_proc/test/PCATest.cpp ${CPP_SOURCES})
# target_compile_definitions(PCATest PRIVATE STATS NDEBUG)
# target_compile_options(PCATest PRIVATE -fopenmp)
# target_include_directories(PCATest
#     PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/algorithms
#     PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
#     PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data_tools/ANNs
#     $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
# )
# target_link_libraries(PCATest PRIVATE -lgomp ZLIB::ZLIB)

# Vamana
add_executable(Vamana ${CMAKE_CURRENT_SOURCE_DIR}/data_tools/ANNs/Vamana.cpp ${CPP_SOURCES})
target_compile_definitions(Vamana PRIVATE STATS NDEBUG)
target_compile_options(Vamana PRIVATE -fopenmp)
target_include_directories(Vamana
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/algorithms
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data_tools/ANNs
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(Vamana PRIVATE -lgomp ZLIB::ZLIB)

# HNSW
add_executable(HNSW ${CMAKE_CURRENT_SOURCE_DIR}/data_tools/ANNs/HNSW/HNSW.cpp ${CPP_SOURCES})
target_compile_definitions(HNSW PRIVATE STATS NDEBUG)
target_compile_options(HNSW PRIVATE -fopenmp)
target_include_directories(HNSW
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/algorithms
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data_tools/ANNs
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data_tools/ANNs/HNSW
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(HNSW PRIVATE -lgomp ZLIB::ZLIB)

# nnDescent
add_executable(nnDescent ${CMAKE_CURRENT_SOURCE_DIR}/data_tools/ANNs/nnDescent/NNDescent.cpp ${CPP_SOURCES})
target_compile_definitions(nnDescent PRIVATE STATS NDEBUG)
target_compile_options(nnDescent PRIVATE -fopenmp)
target_include_directories(nnDescent
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/algorithms
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data_tools/ANNs
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data_tools/ANNs/nnDescent
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(nnDescent PRIVATE -lgomp ZLIB::ZLIB)

# FGIM
add_executable(FGIM ${CMAKE_CURRENT_SOURCE_DIR}/data_tools/ANNs/FGIM/FGIM.cpp ${CPP_SOURCES})
target_compile_definitions(FGIM PRIVATE STATS NDEBUG)
target_compile_options(FGIM PRIVATE -fopenmp)
target_include_directories(FGIM
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/algorithms
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data_tools/ANNs
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(FGIM PRIVATE -lgomp ZLIB::ZLIB)