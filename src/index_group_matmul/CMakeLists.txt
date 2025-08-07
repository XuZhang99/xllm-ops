add_ops_compile_options(
	OP_NAME IndexGroupMatmul
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        op_host/index_group_matmul.cpp
)

# optiling
target_sources(optiling PRIVATE
        op_host/index_group_matmul.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host/index_group_matmul_tiling.h
)


# opproto
target_sources(opsproto PRIVATE
        op_host/index_group_matmul.cpp
)

install(FILES op_kernel/index_group_matmul.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/index_group_matmul.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/index_group_matmul_mlt512.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

