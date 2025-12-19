#include <iostream>
#include <vector>
#include <chrono> 
#include <iomanip> 
#include <cuda_runtime.h>

#include "../include/cifar10_dataset.h"
#include "../include/autoencoder.h"     
#include "../include/gpu_autoencoder.h" 
#include <fstream> 

// =============================================================================
// CÁC HÀM PHỤ TRỢ (SAVE MODEL, MEMORY CHECK)
// =============================================================================

void save_gpu_layer(std::ofstream& file, float* d_data, int size) {
    std::vector<float> h_data(size);
    cudaMemcpy(h_data.data(), d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    file.write(reinterpret_cast<char*>(h_data.data()), size * sizeof(float));
}

void save_gpu_model(GPUAutoencoder& model, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file for save: " << filename << std::endl;
        return;
    }
    
    save_gpu_layer(file, model.d_conv1_w, 256*3*3*3);
    save_gpu_layer(file, model.d_conv1_b, 256);
    save_gpu_layer(file, model.d_conv2_w, 128*256*3*3);
    save_gpu_layer(file, model.d_conv2_b, 128);
    save_gpu_layer(file, model.d_conv3_w, 128*128*3*3);
    save_gpu_layer(file, model.d_conv3_b, 128);
    save_gpu_layer(file, model.d_conv4_w, 256*128*3*3);
    save_gpu_layer(file, model.d_conv4_b, 256);
    save_gpu_layer(file, model.d_conv5_w, 3*256*3*3);
    save_gpu_layer(file, model.d_conv5_b, 3);
    
    std::cout << "\n[SAVE] Model saved to " << filename << std::endl;
    file.close();
}

void print_gpu_memory_usage() {
    size_t free_byte, total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    double free_db = (double)free_byte / (1024.0 * 1024.0);
    double total_db = (double)total_byte / (1024.0 * 1024.0);
    double used_db = total_db - free_db;
    
    std::cout << "   - VRAM Used : " << std::fixed << std::setprecision(2) << used_db << " MB" << std::endl;
    std::cout << "   - VRAM Total: " << total_db << " MB" << std::endl;
}

// =============================================================================
// HÀM CHẠY PHASE 2: NAIVE
// =============================================================================
void run_phase2_training(GPUAutoencoder& gpu_model, CIFAR10Dataset& dataset, int batch_size, int epochs, float lr) {
    std::cout << "==================================================" << std::endl;
    std::cout << "      PHASE 2: NAIVE GPU TRAINING (GLOBAL MEM)    " << std::endl;
    std::cout << "==================================================" << std::endl;

    float *d_batch_data;
    cudaMalloc(&d_batch_data, batch_size * 3 * 32 * 32 * sizeof(float));
    std::vector<float> h_batch_data;
    double total_gpu_time = 0.0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        dataset.shuffle_data();
        int batch_count = 0;
        float epoch_loss = 0.0f;

        while (dataset.get_next_batch(batch_size, h_batch_data)) {
            if (h_batch_data.size() / (3 * 32 * 32) != batch_size) continue;

            // Copy Host -> Device
            cudaMemcpy(d_batch_data, h_batch_data.data(), batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice);
            
            // --- Phase 2 Logic ---
            gpu_model.forward(d_batch_data); // Gọi hàm Naive

            float loss = gpu_model.compute_loss(d_batch_data);
            epoch_loss += loss;
            gpu_model.backward(d_batch_data);
            gpu_model.update(lr);
            batch_count++;
        }
        
        cudaDeviceSynchronize();
        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_sec = std::chrono::duration<double>(epoch_end - epoch_start).count();
        total_gpu_time += epoch_sec;

        std::cout << "Epoch " << std::setw(2) << epoch + 1 << "/" << epochs 
                  << " | Time: " << std::fixed << std::setprecision(2) << epoch_sec << "s"
                  << " | Avg Loss: " << std::setprecision(5) << epoch_loss / batch_count << std::endl;
    }

    double avg_time = total_gpu_time / epochs;
    std::cout << "\n==================================================" << std::endl;
    std::cout << "               GPU RESULT REPORT (PHASE 2)        " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "1. PERFORMANCE:" << std::endl;
    std::cout << "   - Total Time: " << total_gpu_time << " s" << std::endl;
    std::cout << "   - Avg Epoch : " << avg_time << " s" << std::endl;
    std::cout << "\n2. MEMORY:" << std::endl;
    print_gpu_memory_usage();
    std::cout << "==================================================" << std::endl;

    save_gpu_model(gpu_model, "./output/model_gpu_phase2.bin");
    cudaFree(d_batch_data);
}

// =============================================================================
// HÀM CHẠY PHASE 3: OPTIMIZED (SHARED MEMORY)
// =============================================================================
void run_phase3_training(GPUAutoencoder& gpu_model, CIFAR10Dataset& dataset, int batch_size, int epochs, float lr) {
    std::cout << "==================================================" << std::endl;
    std::cout << "      PHASE 3: OPTIMIZED TRAINING (SHARED MEM)    " << std::endl;
    std::cout << "==================================================" << std::endl;

    float *d_batch_data;
    cudaMalloc(&d_batch_data, batch_size * 3 * 32 * 32 * sizeof(float));
    std::vector<float> h_batch_data;
    double total_gpu_time = 0.0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        dataset.shuffle_data();
        int batch_count = 0;
        float epoch_loss = 0.0f;

        while (dataset.get_next_batch(batch_size, h_batch_data)) {
            if (h_batch_data.size() / (3 * 32 * 32) != batch_size) continue;

            // Copy Host -> Device
            cudaMemcpy(d_batch_data, h_batch_data.data(), batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice);
            
            // --- Phase 3 Logic ---
            // Cần copy vào d_input nội bộ của class trước khi gọi kernel tối ưu
            cudaMemcpy(gpu_model.d_input, d_batch_data, batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyDeviceToDevice);
            gpu_model.forward_phase3(); // Gọi hàm Optimized

            float loss = gpu_model.compute_loss(d_batch_data);
            epoch_loss += loss;
            gpu_model.backward(d_batch_data);
            gpu_model.update(lr);
            batch_count++;
        }
        
        cudaDeviceSynchronize();
        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_sec = std::chrono::duration<double>(epoch_end - epoch_start).count();
        total_gpu_time += epoch_sec;

        std::cout << "Epoch " << std::setw(2) << epoch + 1 << "/" << epochs 
                  << " | Time: " << std::fixed << std::setprecision(2) << epoch_sec << "s"
                  << " | Avg Loss: " << std::setprecision(5) << epoch_loss / batch_count << std::endl;
    }

    double avg_time = total_gpu_time / epochs;
    std::cout << "\n==================================================" << std::endl;
    std::cout << "               GPU RESULT REPORT (PHASE 3)        " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "1. PERFORMANCE:" << std::endl;
    std::cout << "   - Total Time: " << total_gpu_time << " s" << std::endl;
    std::cout << "   - Avg Epoch : " << avg_time << " s" << std::endl;
    std::cout << "\n2. MEMORY:" << std::endl;
    print_gpu_memory_usage();
    std::cout << "==================================================" << std::endl;

    save_gpu_model(gpu_model, "./output/model_gpu_phase3.bin");
    cudaFree(d_batch_data);
}

// =============================================================================
// MAIN FUNCTION
// =============================================================================
int main() {
    // --- [CONFIG] CHỌN PHASE ĐỂ CHẠY TẠI ĐÂY ---
    bool RUN_PHASE_3 = true; 
    // true  = Chạy Phase 3 (Optimized)
    // false = Chạy Phase 2 (Naive)

    // 1. Setup Data
    std::string data_path = "./data"; 
    CIFAR10Dataset dataset(data_path);
    dataset.load_data();

    // 2. Config
    int batch_size = 64;       
    int target_epochs = 20;
    float learning_rate = 0.001f;

    std::cout << "[CONFIG] Target Epochs: " << target_epochs << std::endl;
    
    // 3. Init Weights
    std::cout << "[INIT] Generating random weights..." << std::endl;
    Autoencoder cpu_helper; 
    
    std::cout << "[INIT] Booting up GPU..." << std::endl;
    GPUAutoencoder gpu_model(batch_size);

    gpu_model.loadWeights(
        cpu_helper.w1, cpu_helper.b1, cpu_helper.w2, cpu_helper.b2,
        cpu_helper.w3, cpu_helper.b3, cpu_helper.w4, cpu_helper.b4, cpu_helper.w5, cpu_helper.b5
    );

    std::cout << "[INFO] Training Started..." << std::endl;

    // 4. Run Selected Phase
    if (RUN_PHASE_3) {
        run_phase3_training(gpu_model, dataset, batch_size, target_epochs, learning_rate);
    } else {
        run_phase2_training(gpu_model, dataset, batch_size, target_epochs, learning_rate);
    }

    std::cout << "Done." << std::endl;
    return 0;
}