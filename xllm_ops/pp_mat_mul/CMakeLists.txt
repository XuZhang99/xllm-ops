add_ops_compile_options(
	OP_NAME PpMatMul
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        op_host/pp_mat_mul.cpp
)

# optiling
target_sources(optiling PRIVATE
        # op_host/pp_mat_mul.cpp
        op_host/pp_mat_mul_tiling.cpp
        op_host/pp_mat_mul_default.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host/
)


# opproto
target_sources(opsproto PRIVATE
        op_host/pp_mat_mul.cpp
)

install(FILES op_kernel/pp_mat_mul.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)


