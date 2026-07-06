add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
   target_sources(op_host_aclnn PRIVATE
        laser_attention_def.cpp
   )
endif()

add_ops_compile_options(
        OP_NAME LaserAttention
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)


if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE laser_attention ACLNNTYPE aclnn)
endif()
