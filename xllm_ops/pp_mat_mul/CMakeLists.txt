add_ops_compile_options(
	OP_NAME PpMatMul
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Wno-error
)

target_sources(op_host_aclnn PRIVATE
        pp_mat_mul_cpu.cpp
)

# optiling
target_sources(optiling PRIVATE
        # op_host/pp_mat_mul.cpp
        pp_mat_mul_tiling.cpp
        pp_mat_mul_default.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/
)


# opproto
target_sources(opsproto PRIVATE
       pp_mat_mul_cpu.cpp
)

install(FILES pp_mat_mul.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)


