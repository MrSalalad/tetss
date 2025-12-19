#include "../include/kernels.cuh"
#include <cstdio>

// Cấu hình kích thước Tile (phải khớp với blockDim trong gpu_autoencoder.cu)
#define TILE_WIDTH 16
#define K_SIZE 3
// Kích thước mảng Shared Memory cần thiết: (16 pixel output + 2 pixel viền)
#define SHARED_WIDTH (TILE_WIDTH + K_SIZE - 1) 

namespace Phase3Kernels {

    // =================================================================================
    // OPTIMIZED CONVOLUTION: CORE LOGIC PHASE 2 + SHARED MEMORY
    // =================================================================================
    __global__ void conv2d_shared_mem_kernel(const float* __restrict__ input, 
                                             float* __restrict__ output, 
                                             const float* __restrict__ weights, 
                                             const float* __restrict__ bias,
                                             int batch_size, int in_c, int out_c, int in_h, int in_w, 
                                             int out_h, int out_w, int k_size, int padding, int stride) 
    {
        // 1. XÁC ĐỊNH TỌA ĐỘ (Giữ nguyên logic mapping cơ bản)
        int tx = threadIdx.x; 
        int ty = threadIdx.y; 
        
        // Tọa độ Output trên ảnh toàn cục
        int row_o = blockIdx.y * TILE_WIDTH + ty;
        int col_o = blockIdx.x * TILE_WIDTH + tx;
        
        // Mapping Batch và Output Channel từ Grid Z
        int b = blockIdx.z / out_c;
        int oc = blockIdx.z % out_c;

        // 2. SHARED MEMORY (Phần nâng cấp)
        // Thay vì đọc Global Input, ta dùng mảng này làm bộ đệm tốc độ cao
        __shared__ float s_input[SHARED_WIDTH][SHARED_WIDTH];

        float sum = 0.0f;

        // 3. VÒNG LẶP CHÍNH (Giữ nguyên logic duyệt qua Input Channel)
        for (int ic = 0; ic < in_c; ++ic) {

            // --- BƯỚC A: LOAD DỮ LIỆU (Kế thừa logic check biên của Phase 2) ---
            
            // Xác định gốc tọa độ của Tile trên Input Global (bao gồm cả padding ảo)
            int input_start_row = blockIdx.y * TILE_WIDTH - padding;
            int input_start_col = blockIdx.x * TILE_WIDTH - padding;

            // Mỗi thread load một phần tử vào Shared Memory.
            // Do Tile Input (18x18=324) lớn hơn số thread (16x16=256), ta dùng vòng lặp để load hết.
            int tid = ty * TILE_WIDTH + tx; 

            for (int i = tid; i < SHARED_WIDTH * SHARED_WIDTH; i += TILE_WIDTH * TILE_WIDTH) {
                int s_r = i / SHARED_WIDTH; // Row trong Shared
                int s_c = i % SHARED_WIDTH; // Col trong Shared
                
                // Tọa độ thực tế trên Global Input
                int in_r = input_start_row + s_r;
                int in_c_loc = input_start_col + s_c;

                // [LOGIC GỐC PHASE 2]: Kiểm tra padding
                if (in_r >= 0 && in_r < in_h && in_c_loc >= 0 && in_c_loc < in_w) {
                    // Công thức Index y hệt Phase 2
                    int in_idx = b * (in_c * in_h * in_w) + ic * (in_h * in_w) + in_r * in_w + in_c_loc;
                    s_input[s_r][s_c] = input[in_idx];
                } else {
                    s_input[s_r][s_c] = 0.0f; // Padding giá trị 0
                }
            }

            // Đồng bộ: Đợi tất cả thread load xong mới được tính
            __syncthreads(); 

            // --- BƯỚC B: TÍNH TOÁN (Logic nhân chập y hệt, chỉ đổi nguồn dữ liệu) ---
            
            // Chỉ những thread nằm trong phạm vi Output mới tính
            if (ty < TILE_WIDTH && tx < TILE_WIDTH && row_o < out_h && col_o < out_w) {
                
                // [LOGIC GỐC PHASE 2]: Duyệt Kernel 3x3
                for (int kh = 0; kh < k_size; ++kh) {
                    for (int kw = 0; kw < k_size; ++kw) {
                        
                        // Lấy Weight (Y hệt Phase 2)
                        int w_idx = oc * (in_c * k_size * k_size) + ic * (k_size * k_size) + kh * k_size + kw;
                        
                        // Lấy Input (KHÁC BIỆT DUY NHẤT: Lấy từ s_input thay vì input)
                        // Vị trí tương ứng trong shared memory là (ty + kh, tx + kw)
                        sum += s_input[ty + kh][tx + kw] * weights[w_idx];
                    }
                }
            }
            
            // Đồng bộ: Đợi tính xong trước khi load channel tiếp theo đè lên s_input
            __syncthreads();
        }

        // 4. GHI KẾT QUẢ (Giữ nguyên logic cộng Bias của Phase 2)
        if (ty < TILE_WIDTH && tx < TILE_WIDTH && row_o < out_h && col_o < out_w) {
            if (bias != nullptr) {
                sum += bias[oc];
            }
            int out_idx = b * (out_c * out_h * out_w) + oc * (out_h * out_w) + row_o * out_w + col_o;
            output[out_idx] = sum;
        }
    }
}