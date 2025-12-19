// File: src/kernels_phase3.cu
#include "../include/kernels.cuh"
#include <cstdio>

// Định nghĩa kích thước Tile
#define TILE_WIDTH 16   // Kích thước đầu ra của mỗi block (16x16 pixel)
#define K_SIZE 3        // Kernel size (3x3)
#define HALO_SIZE 1     // Padding viền (3/2 = 1)
// Kích thước Shared Memory cần thiết để tính được 16x16 output:
// Cần load (16 + 2*1) = 18x18 pixel từ Input
#define SHARED_WIDTH (TILE_WIDTH + 2 * HALO_SIZE) 

namespace Phase3Kernels {

    // =================================================================================
    // CATEGORY 1: MEMORY OPTIMIZATION (SHARED MEMORY TILING)
    // =================================================================================
    /*
       Ý tưởng tối ưu:
       1. Thay vì mỗi thread tự đọc 9 pixel từ Global Memory (rất chậm).
       2. Cả Block sẽ hợp sức load 1 mảng Input Tile (18x18) vào Shared Memory (nhanh như L1 Cache).
       3. Tính toán Convolution bằng cách đọc từ Shared Memory.
    */
    __global__ void conv2d_shared_mem_kernel(const float* __restrict__ input, 
                                             float* __restrict__ output, 
                                             const float* __restrict__ weights, 
                                             const float* __restrict__ bias,
                                             int batch_size, int in_c, int out_c, int in_h, int in_w, 
                                             int out_h, int out_w, int k_size, int padding, int stride) 
    {
        // 1. Xác định vị trí Pixel Output mà thread này phụ trách
        int tx = threadIdx.x; // 0..15
        int ty = threadIdx.y; // 0..15
        
        int row_o = blockIdx.y * TILE_WIDTH + ty;
        int col_o = blockIdx.x * TILE_WIDTH + tx;
        
        // Map Grid Z sang Batch và Output Channel
        int b = blockIdx.z / out_c;
        int oc = blockIdx.z % out_c;

        // 2. Khai báo Shared Memory
        // Kích thước: [18][18]
        __shared__ float s_input[SHARED_WIDTH][SHARED_WIDTH];

        float value = 0.0f;

        // 3. Loop qua từng Input Channel (in_c)
        // Vì Shared Memory có hạn, ta xử lý từng channel (hoặc nhóm channel) tuần tự
        for (int ic = 0; ic < in_c; ++ic) {

            // --- GIAI ĐOẠN 1: LOAD DỮ LIỆU TỪ GLOBAL VÀO SHARED MEMORY ---
            // Cần load vùng ảnh Input kích thước 18x18 bao quanh vùng Output 16x16
            // Góc trên trái của vùng Input cần load (tính cả padding ảo)
            int input_start_row = blockIdx.y * TILE_WIDTH - padding;
            int input_start_col = blockIdx.x * TILE_WIDTH - padding;

            // Mapping: Block ta có 256 threads (16x16), cần load 324 phần tử (18x18).
            // Ta dùng vòng lặp để thread phủ hết mảng shared memory.
            int tid = ty * TILE_WIDTH + tx; // Thread ID tuyến tính (0-255)
            
            for (int i = tid; i < SHARED_WIDTH * SHARED_WIDTH; i += TILE_WIDTH * TILE_WIDTH) {
                int s_r = i / SHARED_WIDTH; // Row trong Shared Mem
                int s_c = i % SHARED_WIDTH; // Col trong Shared Mem
                
                int in_r = input_start_row + s_r;
                int in_c_loc = input_start_col + s_c;

                // Kiểm tra biên (Padding Zero)
                if (in_r >= 0 && in_r < in_h && in_c_loc >= 0 && in_c_loc < in_w) {
                    // Index Global: Batch -> Channel -> Row -> Col
                    int in_idx = b * (in_c * in_h * in_w) + ic * (in_h * in_w) + in_r * in_w + in_c_loc;
                    s_input[s_r][s_c] = input[in_idx];
                } else {
                    s_input[s_r][s_c] = 0.0f;
                }
            }

            __syncthreads(); // BẮT BUỘC: Đợi tất cả thread load xong Tile hiện tại

            // --- GIAI ĐOẠN 2: TÍNH TOÁN CONVOLUTION ---
            if (ty < TILE_WIDTH && tx < TILE_WIDTH && row_o < out_h && col_o < out_w) {
                // Duyệt qua Kernel 3x3
                for (int i = 0; i < k_size; ++i) {
                    for (int j = 0; j < k_size; ++j) {
                        // Lấy weight từ Global Memory
                        int w_idx = oc * (in_c * k_size * k_size) + ic * (k_size * k_size) + i * k_size + j;
                        
                        // Lấy input từ Shared Memory (Cực nhanh)
                        // Vị trí (ty, tx) trong Output tương ứng với (ty, tx) trong Shared Input 
                        // nhưng lệch bởi vòng lặp kernel (i,j)
                        value += s_input[ty + i][tx + j] * weights[w_idx];
                    }
                }
            }
            __syncthreads(); // Đợi tính xong trước khi load channel tiếp theo vào Shared Mem
        }

        // 4. Ghi kết quả ra Global Memory
        if (ty < TILE_WIDTH && tx < TILE_WIDTH && row_o < out_h && col_o < out_w) {
            if (bias != nullptr) {
                value += bias[oc];
            }
            int out_idx = b * (out_c * out_h * out_w) + oc * (out_h * out_w) + row_o * out_w + col_o;
            output[out_idx] = value;
        }
    }
}