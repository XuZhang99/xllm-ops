add_ops_compile_options(
	OP_NAME MultiLatentAttention
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
)

target_sources(op_host_aclnn PRIVATE
        op_host/multi_latent_attention.cpp
)

# optiling
target_sources(optiling PRIVATE
        op_host/multi_latent_attention.cpp
        op_host/math.cpp
        op_host/multi_latent_attention_tiling_impl.cpp
        op_host/multi_latent_attention_tiling_dependency.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host/
)


# opproto
target_sources(opsproto PRIVATE
        op_host/multi_latent_attention.cpp
)

install(FILES op_kernel/multi_latent_attention.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)




