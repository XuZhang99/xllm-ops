add_ops_compile_options(
    OP_NAME RecConstrainedTopK
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Wno-error
)

target_sources(op_host_aclnn PRIVATE
    rec_constrained_top_k_cpu.cpp
)

target_sources(optiling PRIVATE
    rec_constrained_top_k_cpu.cpp
)

target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
    rec_constrained_top_k_cpu.cpp
)

install(FILES rec_constrained_top_k.cpp
    DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES rec_constrained_top_k.h
    DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
