@echo off
echo ==================================================
echo       BUILDING PHASE 3: OPTIMIZED MEMORY
echo ==================================================

REM 1. Dọn dẹp file cũ (nếu có)
if exist gpu_phase3.exe del gpu_phase3.exe

REM 2. Lệnh biên dịch (Compile)
REM Cần compile tất cả các file .cu và .cpp liên quan:
REM - main_gpu.cu: File chính
REM - gpu_autoencoder.cu: Class quản lý GPU
REM - kernels_naive.cu: Chứa các kernel phụ trợ (ReLU, Pool, Loss...) vẫn dùng lại
REM - kernels_phase3.cu: Chứa kernel Convolution tối ưu (Shared Memory)
REM - cifar10_dataset.cpp: Đọc dữ liệu
REM - autoencoder.cpp: Sinh số ngẫu nhiên ban đầu

echo Compiling sources...
nvcc -O3 -I ./include main_gpu.cu src/gpu/gpu_autoencoder.cu src/gpu/kernels_naive.cu src/gpu/kernels_phase3.cu src/cifar10_dataset.cpp src/autoencoder.cpp -o gpu_phase3.exe

REM 3. Kiểm tra lỗi biên dịch
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Compilation Failed! Please check the errors above.
    goto end
)

echo.
echo [SUCCESS] Compilation finished. Created gpu_phase3.exe
echo.

REM 4. Chạy chương trình
echo ==================================================
echo            RUNNING GPU TRAINING (LOCAL)
echo ==================================================
gpu_phase3.exe

:end
pause