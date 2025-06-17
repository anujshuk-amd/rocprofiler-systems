// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc.
// [License text omitted for brevity]


#if defined(USE_ROCTRACER_ROCTX)
#    include <roctracer/roctx.h>
#else
#    include <rocprofiler-sdk-roctx/roctx.h>
#endif

#include <hip/hip_runtime.h>

#if defined(USE_MPI)
#    include <mpi.h>
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>
#include <sstream>
#include <cstddef>
// HIP and ROCm profiling headers
#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include <hsa/hsa.h>

#if defined(USE_MPI)
#    include <mpi.h>
#endif
void run(int rank, int tid, int ndevice, int argc, char** argv);

// namespace
// {
//     using auto_lock_t = std::unique_lock<std::mutex>;
//     auto print_lock = std::mutex{};
// }
namespace
{
    using auto_lock_t                      = std::unique_lock<std::mutex>;
    auto               print_lock          = std::mutex{};
    size_t             nthreads            = 2;
    size_t             nitr                = 500;
    size_t             nsync               = 10;
    constexpr unsigned shared_mem_tile_dim = 32;

    // void
    // check_hip_error(void);

    void
    verify(int* in, int* out, int M, int N);
}  // namespace
// Define HIP_API_CALL macro for error handling.
#define HIP_API_CALL(CALL)                                                                         \
    {                                                                                              \
        hipError_t error_ = (CALL);                                                                \
        if (error_ != hipSuccess)                                                                  \
        {                                                                                          \
            auto _hip_api_print_lk = auto_lock_t{print_lock};                                      \
            fprintf(stderr,                                                                        \
                    "%s:%d :: HIP error : %s\n",                                                   \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    hipGetErrorString(error_));                                                    \
            throw std::runtime_error("hip_api_call");                                              \
        }                                                                                          \
    }



// HIP Kernel Function
__global__ void hipKernelLaunch(int* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] += 1;
}
// Function to execute GPU workload with ROCTx profiling
void gpu_workload() 
{
    // Start a profiling range and push a sub-range for launching the kernel.
    uint64_t rangeId = roctxRangeStart("Roctx: GPU Compute Phase");
    roctxRangePush("Roctx: Launching HIP Kernel");
    
    const int N = 256;
    int *d_data = nullptr;
    
    // Allocate device memory
    
    HIP_API_CALL(hipMalloc(&d_data, N * sizeof(int)));

    // Launch the kernel
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipKernelLaunch), dim3(1), dim3(N), 0, 0, d_data);

    // Wait for GPU to finish
    HIP_API_CALL(hipDeviceSynchronize());

    std::cout << "Kernel execution completed!" << std::endl;

    // Free device memory
    HIP_API_CALL(hipFree(d_data));

    // Pop the sub-range and stop the profiling range
    roctxRangePop();
    roctxRangeStop(rangeId);
}

// Function executed in a separate thread with ROCTx annotations.
void roctx_thread_function() {
    roctxNameOsThread("Roctx: WorkerThread");
    roctxMark("Roctx: Thread Execution Started");
    gpu_workload();
    roctxMark("Roctx: Thread Execution Completed");
    std::this_thread::sleep_for(std::chrono::seconds(10));
}



void
run_profiling()
{
    std::this_thread::sleep_for(std::chrono::seconds(10));

    // Insert a marker before the GPU workload
    roctxMark("Roctx: Starting profiling");

	// Label HIP device and stream
    int deviceId{0};
    HIP_API_CALL(hipGetDevice(&deviceId));
    roctxNameHipDevice("Roctx: AMD GPU Device Id :", deviceId);

    hipStream_t stream = {};
    HIP_API_CALL(hipStreamCreate(&stream));
    roctxNameHipStream("Roctx: HIP Compute Stream :", stream);

    // Start a nested profiling range.
    roctxRangePush("Roctx: run_profiling execution marking");
    // End the nested profiling range.
    roctxRangePop();
    // Execute GPU workload
    gpu_workload();

    // Pause profiling steps using ROCTx APIs.
    roctx_thread_id_t roctx_tid{};      // Thread identifier structure
    roctxGetThreadId(&roctx_tid);
    
    // Set names for OS thread, HSA agent, HIP device and stream.
    roctxNameOsThread(std::to_string(roctx_tid).c_str());
    // Prepare an hsa_agent_t with roctx thread id as a handle (example usage):
    hsa_agent_t hsa_agent = { .handle = roctx_tid };
    roctxNameHsaAgent("Roctx: hsa_agent", &hsa_agent);
    roctxNameHipDevice("Roctx: hip_device", 0);
    auto* hip_stream = hipStream_t{};
    roctxNameHipStream("Roctx: hip_stream", hip_stream);

    // Pause ROCTx profiling for the current thread.
    roctxProfilerPause(roctx_tid);
    roctxMark(__FUNCTION__);



    // Resume ROCTx profiling.
    roctxProfilerResume(roctx_tid);



    // Insert a marker after execution of workload.
    roctxMark("Roctx: Finished GPU workload");

    std::cout << "Roctx:Profiling completed!" << std::endl;
    HIP_API_CALL(hipStreamDestroy(stream));



}

void test1()
{
    roctxMark("before hipLaunchKernel");
    int rangeId = roctxRangeStart("hipLaunchKernel range");
    roctxRangePush("hipLaunchKernel");

    // Launching kernel from host    
    const int N = 256;
    int *d_data = nullptr;
    // Allocate device memory
    HIP_API_CALL(hipMalloc(&d_data, N * sizeof(int)));
    // Launch the kernel
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipKernelLaunch), dim3(1), dim3(N), 0, 0, d_data);
    roctxMark("after hipLaunchKernel");

    // Memory transfer from device to host
    roctxRangePush("hipMemcpy");
    // Allocate and copy vectors to device memory.
    // Allocate and copy vectors to device memory.
    float* d_x{};
    size_t size_bytes = 256; 
    std::vector<float> x(256, 1.0f);
    // Allocate memory for both vectors on the device
    HIP_API_CALL(hipMalloc(&d_x, size_bytes));
    // Copy data from host to device
    HIP_API_CALL(hipMemcpy(d_x, x.data(), size_bytes, hipMemcpyHostToDevice));
    roctxRangePop();  // for "hipMemcpy"
    roctxRangePop();  // for "hipLaunchKernel"
    roctxRangeStop(rangeId);
}

int main22() {
    
    std::thread t1(run_profiling);    
    // Start a separate thread executing additional profiling-annotated work.
    std::thread t2(roctx_thread_function);

    t1.join();
    t2.join();
    test1();
    return 0;
}
//===================
void run(int rank, int tid, int devid, int argc, char** argv)
{
    auto roctx_run_id = roctxRangeStart("run");

    const auto mark = [rank, tid, devid](std::string_view suffix) {
        auto _ss = std::stringstream{};
        _ss << "run/rank-" << rank << "/thread-" << tid << "/device-" << devid << "/" << suffix;
        roctxMark(_ss.str().c_str());
    };

    mark("begin");

    constexpr unsigned int M = 4960 * 2;
    constexpr unsigned int N = 4960 * 2;

    if(argc > 2) nitr = atoll(argv[2]);
    if(argc > 3) nsync = atoll(argv[3]);

    hipStream_t stream = {};

    printf("[roctxAnuj] Rank %i, thread %i assigned to device %i\n", rank, tid, devid);
    HIP_API_CALL(hipSetDevice(devid));
    HIP_API_CALL(hipStreamCreate(&stream));

    auto_lock_t _lk{print_lock};
    std::cout << "[roctxAnuj][" << rank << "][" << tid << "] M: " << M << " N: " << N << std::endl;
    _lk.unlock();

    std::default_random_engine         _engine{std::random_device{}() * (rank + 1) * (tid + 1)};
    std::uniform_int_distribution<int> _dist{0, 1000};

    size_t size       = sizeof(int) * M * N;
    int*   inp_matrix = new int[size];
    int*   out_matrix = new int[size];
    for(size_t i = 0; i < M * N; i++)
    {
        inp_matrix[i] = _dist(_engine);
        out_matrix[i] = 0;
    }
    int* in  = nullptr;
    int* out = nullptr;

    // lock during malloc to get more accurate memory info
    {
        _lk.lock();
        constexpr auto MiB           = (1024UL * 1024UL);
        size_t         free_gpu_mem  = 0;
        size_t         total_gpu_mem = 0;

        HIP_API_CALL(hipMemGetInfo(&free_gpu_mem, &total_gpu_mem));
        free_gpu_mem /= MiB;
        total_gpu_mem /= MiB;

        std::cout << "[roctxAnuj][" << rank << "][" << tid
                  << "] Available GPU memory (MiB): " << std::setw(6) << free_gpu_mem << " / "
                  << std::setw(6) << total_gpu_mem << std::endl;

        HIP_API_CALL(hipMallocAsync(&in, size, stream));
        HIP_API_CALL(hipMallocAsync(&out, size, stream));

        _lk.unlock();
    }

    // aleks:
    // Declare and create the event.
    hipEvent_t stop;
    HIP_API_CALL(hipEventCreate(&stop));

    HIP_API_CALL(hipMemsetAsync(in, 0, size, stream));
    HIP_API_CALL(hipMemsetAsync(out, 0, size, stream));
    HIP_API_CALL(hipMemcpyAsync(in, inp_matrix, size, hipMemcpyHostToDevice, stream));
    HIP_API_CALL(hipStreamSynchronize(stream));

    dim3 grid(M / 32, N / 32, 1);
    dim3 block(32, 32, 1);  // roctxAnuj

    auto t1 = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < nitr; ++i)
    {
        roctxRangePush("run/iteration");
       // HIP_API_CALL<<<grid, block, 0, stream>>>(in, out, M, N);
        //check_hip_error();
        if(i % nsync == (nsync - 1))
        {
            roctxRangePush("run/iteration/sync");
            HIP_API_CALL(hipStreamSynchronize(stream));
            roctxRangePop();
        }
        roctxRangePop();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    HIP_API_CALL(hipStreamSynchronize(stream));
    HIP_API_CALL(hipMemcpyAsync(out_matrix, out, size, hipMemcpyDeviceToHost, stream));
    double time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    float  GB   = (float) size * nitr * 2 / (1 << 30);

    // aleks:
    // Record the stop event.
    printf("aleks: deploy_multiple_stream %s:%d: Before hipEventRecord\n", __FILE__, __LINE__);
    HIP_API_CALL(hipEventRecord(stop, stream));
    printf("aleks: deploy_multiple_stream %s:%d: After hipEventRecord\n", __FILE__, __LINE__);

    print_lock.lock();
    std::cout << "[roctxAnuj][" << rank << "][" << tid << "] Runtime of roctxAnuj is " << time
              << " sec\n";
    std::cout << "[roctxAnuj][" << rank << "][" << tid
              << "] The average performance of roctxAnuj is " << GB / time << " GBytes/sec"
              << std::endl;
    print_lock.unlock();

    HIP_API_CALL(hipStreamSynchronize(stream));

    HIP_API_CALL(hipEventSynchronize(stop));


    HIP_API_CALL(hipFreeAsync(in, stream));
    HIP_API_CALL(hipFreeAsync(out, stream));

    HIP_API_CALL(hipStreamSynchronize(stream));
    HIP_API_CALL(hipStreamDestroy(stream));

    delete[] inp_matrix;
    delete[] out_matrix;

    mark("end");

    roctxRangeStop(roctx_run_id);
}

int
main(int argc, char** argv)
{
    int rank = 0;
    int size = 1;
    for(int i = 1; i < argc; ++i)
    {
        auto _arg = std::string{argv[i]};
        if(_arg == "?" || _arg == "-h" || _arg == "--help")
        {
            fprintf(stderr,
                    "usage: roctxAnuj [NUM_THREADS (%zu)] [NUM_ITERATION (%zu)] "
                    "[SYNC_EVERY_N_ITERATIONS (%zu)]\n",
                    nthreads,
                    nitr,
                    nsync);
            exit(EXIT_SUCCESS);
        }
    }
    if(argc > 1) nthreads = atoll(argv[1]);
    if(argc > 2) nitr = atoll(argv[2]);
    if(argc > 3) nsync = atoll(argv[3]);

    printf("[roctxAnuj] Number of threads: %zu\n", nthreads);
    printf("[roctxAnuj] Number of iterations: %zu\n", nitr);
    printf("[roctxAnuj] Syncing every %zu iterations\n", nsync);

    // this is a temporary workaround in omnitrace when HIP + MPI is enabled
    int ndevice = 0;
    HIP_API_CALL(hipGetDeviceCount(&ndevice));
    printf("[roctxAnuj] Number of devices found: %i\n", ndevice);
    auto devids = std::vector<int>{};
    devids.resize(size * nthreads, 0);
    int devid = 0;
    for(size_t i = 0; i < nthreads; ++i)
    {
        for(int j = 0; j < size; ++j)
        {
            auto idx       = (j * nthreads) + i;
            devids.at(idx) = devid++ % ndevice;
        }
    }
    auto devid_offset = (rank * nthreads);
    auto _threads     = std::vector<std::thread>{};
    for(size_t i = 1; i < nthreads; ++i)
        _threads.emplace_back(run, rank, i, devids.at(devid_offset + i), argc, argv);
    run(rank, 0, devids.at(devid_offset + 0), argc, argv);
    for(auto& itr : _threads)
        itr.join();

    // this is a temporary workaround in omnitrace when HIP + MPI is enabled
     ndevice = 0;
    HIP_API_CALL(hipGetDeviceCount(&ndevice));
    printf("[roctxAnuj] Number of devices found: %i\n", ndevice);
    for(int i = 0; i < ndevice; ++i)
    {
        HIP_API_CALL(hipSetDevice(i));
        HIP_API_CALL(hipDeviceSynchronize());
    }

    if(rank == 0)
    {
        for(int i = 0; i < ndevice; ++i)
        {
            HIP_API_CALL(hipSetDevice(i));
            HIP_API_CALL(hipDeviceReset());
        }
    }
        roctxRangePush("Processing Loop");
    for (int i = 0; i < 1000; i++) {
        volatile int temp = i * i; // Prevent compiler optimization
    }
    roctxRangePop();

    main22();
    return 0;
}
