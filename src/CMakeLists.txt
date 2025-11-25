add_executable(${PROJECT_NAME} main.cpp testCuda.cu
        heat.cu colormap.cu)

set_target_properties(Heat_Transfer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${PROJECT_NAME} PRIVATE )
target_link_libraries(${PROJECT_NAME} PRIVATE vendor)