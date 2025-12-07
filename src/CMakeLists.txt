add_executable(${PROJECT_NAME} main.cpp testCuda.cu
        heat.cu)

set_target_properties(Heat_Transfer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${PROJECT_NAME} PRIVATE )
target_link_libraries(${PROJECT_NAME} PRIVATE vendor)

# Add the timing test executable
add_executable(timing_test timing_test.cu heat.cu)
set_target_properties(timing_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add tests
add_test(NAME timing_test COMMAND timing_test)
add_test(NAME main_app_smoke_test COMMAND ${PROJECT_NAME})
set_tests_properties(main_app_smoke_test PROPERTIES TIMEOUT 5)