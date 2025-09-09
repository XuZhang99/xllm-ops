#include "kernel_operator.h"
#include "multi_latent_attention.h"
#include "lib/matmul_intf.h"

extern "C" __global__ __aicore__ void multi_latent_attention(GM_ADDR query, GM_ADDR queryRope, GM_ADDR kvCache,
                                                             GM_ADDR kvCacheRope, GM_ADDR block_tables,
                                                             GM_ADDR contextLens, GM_ADDR mask, GM_ADDR qSeqlen,
                                                             GM_ADDR qkDescale, GM_ADDR pvDescale, GM_ADDR attenOut,
                                                             GM_ADDR lseOut, GM_ADDR workspace, GM_ADDR tiling) {
    AscendC::printf("TestHala entering kernel function multi_latent_attention");
    // tiling数据处理，开头的5个是5个workspace的大小：uint64_t*5
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    __gm__ uint64_t *workspaceParam = reinterpret_cast<__gm__ uint64_t *>(tiling);
    GM_ADDR s_gm = usrWorkspace;
    GM_ADDR s_rope_out_gm = s_gm + workspaceParam[0];
    GM_ADDR p_gm = s_rope_out_gm + workspaceParam[1];
    GM_ADDR o_tmp_gm = p_gm + workspaceParam[2];
    GM_ADDR go_gm = o_tmp_gm + workspaceParam[3];
    
    AscendC::printf("TestHala workspaceParam[0]=%lu\n", workspaceParam[0]);
    AscendC::printf("TestHala workspaceParam[1]=%lu\n", workspaceParam[1]);
    AscendC::printf("TestHala workspaceParam[2]=%lu\n", workspaceParam[2]);
    AscendC::printf("TestHala workspaceParam[3]=%lu\n", workspaceParam[3]);
    AscendC::printf("TestHala workspaceParam[4]=%lu\n", workspaceParam[4]);
    // contextLens,qSeqlen参数在当前版本未使用
    // tiling的真实地址需要向后跳转5个uint64_t类型数据
    mla(query, queryRope, kvCache, kvCacheRope, block_tables, mask, qkDescale, pvDescale, attenOut, lseOut,
            s_gm, s_rope_out_gm, p_gm, o_tmp_gm, go_gm, tiling + sizeof(uint64_t) * 5);
}

/*
    __gm__ uint8_t *__restrict__ q_gm,
    __gm__ uint8_t *__restrict__ q_rope_gm,
    __gm__ uint8_t *__restrict__ ctkv_gm,
    __gm__ uint8_t *__restrict__ ctkv_rope_gm,
    __gm__ uint8_t *__restrict__ block_tables_gm,
    __gm__ uint8_t *__restrict__ mask_gm,
    __gm__ uint8_t *__restrict__ deq_qk_gm,
    __gm__ uint8_t *__restrict__ deq_pv_gm,
    __gm__ uint8_t *__restrict__ o_gm,
    __gm__ uint8_t *__restrict__ lse_gm,
    __gm__ uint8_t *__restrict__ s_gm,
    __gm__ uint8_t *__restrict__ s_rope_out_gm,
    __gm__ uint8_t *__restrict__ p_gm,
    __gm__ uint8_t *__restrict__ o_tmp_gm,
    __gm__ uint8_t *__restrict__ go_gm,
    __gm__ uint8_t *__restrict__ tiling_para_gm)
 */
