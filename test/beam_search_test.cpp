#include <gtest/gtest.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "beam_search.h"
#include "base/utils_tensor.h"

// class BeamSearchTest : public ::testing::Test {
// protected:
//     int32_t deviceId = 15;
//     aclrtStream stream1;
    
//     void SetUp() override {
//         utils::initialize_acl(deviceId, &stream1);
//     }

//     void TearDown() override {
//         aclrtDestroyStream(stream1);
//         aclrtResetDevice(deviceId);
//         aclFinalize();
//     }
// };

// TEST_F(BeamSearchTest, BasicMatmulCorrectness) {
//     beam_search::BeamSearchBase inputs(2, 2, 2, 2);
//     inputs.create_torch_tensors();
//     // beam_search::BeamSearchTorch op_torch(inputs);
//     // op_torch.process();
//     // print torch values
//     printf("torch value:\n");
//     {
//         std::ostringstream oss; oss << inputs.token_ids; printf("token_ids:\n%s\n", oss.str().c_str());
//     }
//     {
//         std::ostringstream oss; oss << inputs.log_probs; printf("log_probs:\n%s\n", oss.str().c_str());
//     }
//     {
//         std::ostringstream oss; oss << inputs.top_tokens; printf("top_tokens:\n%s\n", oss.str().c_str());
//     }
//     {
//         std::ostringstream oss; oss << inputs.top_probs; printf("top_probs:\n%s\n", oss.str().c_str());
//     }
//     // {
//     //     std::ostringstream oss; oss << inputs.output_token_ids_torch; printf("output_token_ids_torch:\n%s\n", oss.str().c_str());
//     // }
//     beam_search::BeamSearchOp op_cann(inputs);
//     op_cann.process(stream1);
//     // Ensure NPU ops are finished before reading back
//     aclrtSynchronizeStream(stream1);
//     // Compare on CPU to avoid device mismatch
//     // auto out_torch_cpu = inputs.output_token_ids_torch.to(torch::kCPU);
//     // auto out_op_cpu = inputs.output_token_ids_op.to(torch::kCPU);
//     // std::cout << "out_op_cpu:\n" << out_op_cpu << std::endl;

//     // std::cout << "less than 1e-5: " << (out_torch_cpu - out_op_cpu).abs().max().item<float>() << std::endl;
//     // EXPECT_TRUE(torch::equal(out_torch_cpu, out_op_cpu))
//     //     << "Custom op output does not match native output";
//     op_cann.destroy_tensors();
// }

int main() {
    beam_search::BeamSearchBase inputs(2, 2, 2, 2);
    inputs.create_torch_tensors();
    beam_search::BeamSearchTorch op_torch(inputs);
    op_torch.process();
    cout<<"op_torch done"<<endl;
    cout<< "output_token_ids torch:\n" << inputs.output_token_ids_torch << endl;
    int deviceId = 15;
    aclrtStream stream1;
    utils::initialize_acl(deviceId, &stream1);
    printf("torch value:\n");
    {
        std::ostringstream oss; oss << inputs.token_ids; printf("token_ids:\n%s\n", oss.str().c_str());
    }
    {
        std::ostringstream oss; oss << inputs.log_probs; printf("log_probs:\n%s\n", oss.str().c_str());
    }
    {
        std::ostringstream oss; oss << inputs.top_tokens; printf("top_tokens:\n%s\n", oss.str().c_str());
    }
    {
        std::ostringstream oss; oss << inputs.top_probs; printf("top_probs:\n%s\n", oss.str().c_str());
    }
    beam_search::BeamSearchOp op_cann(inputs);
    op_cann.process(stream1);
    aclrtSynchronizeStream(stream1);
    
    cout<<"op_cann done"<<endl;
    cout<< "output_token_ids:\n" << inputs.output_token_ids_op << endl;
    auto out_torch_cpu = inputs.output_token_ids_torch.to(torch::kCPU);
    auto out_op_cpu = inputs.output_token_ids_op.to(torch::kCPU);
    cout<< "equal: " << torch::equal(out_torch_cpu, out_op_cpu) << endl;
    op_cann.destroy_tensors();
    aclrtDestroyStream(stream1);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
// TEST_F(BeamSearchTest, DifferentSizes) {
//     std::vector<std::tuple<int, int, int, int>> test_sizes = {
//         {2, 2, 2, 2},
//         {2, 2, 2, 2},
//         {2, 2, 2, 2}
//     };
//     for (auto [beam_width, top_k, request_num, sequence_length] : test_sizes) {
//         beam_search::BeamSearchBase inputs(beam_width, top_k, request_num, sequence_length);
//         inputs.create_torch_tensors();
//         beam_search::BeamSearchTorch op_torch(inputs);
//         op_torch.process();
//         beam_search::BeamSearchOp op_cann(inputs);
//         op_cann.process(stream1);
//         EXPECT_TRUE(torch::equal(inputs.output_token_ids_torch, inputs.output_token_ids_op))
//             << "Failed for size: beam_width=" << beam_width << ", top_k=" << top_k << ", request_num=" << request_num << ", sequence_length=" << sequence_length;
//         op_cann.destroy_tensors();
//     }
// }