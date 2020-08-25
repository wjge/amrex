#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <cuda_profiler_api.h>

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    const int N = 460000000;
{
    Gpu::DeviceVector<Real> x_d(N);

    auto x_d_ptr = x_d.dataPtr();

    //warm up
    for(int j=0;j<10;j++)
    {
        AMREX_FOR_1D (N,i,{double dummy=123.456; x_d_ptr[i]=dummy+123.456*dummy;});
    }

    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<float> fsec;

    Gpu::Device::synchronize();
    auto start_clock = Time::now();

    cudaProfilerStart();

    for(int j=0;j<10;j++)
    {
        AMREX_FOR_1D (N,i,{double dummy=123.456; x_d_ptr[i]=dummy+123.456*dummy;});
    }

    Gpu::Device::synchronize();

    cudaProfilerStop();

    auto finish_clock = Time::now();
    fsec fs = finish_clock - start_clock;
    std::cout<<"time taken for amrex_parallel_for (msecs):" << fs.count()*1e3 << std::endl;
}
    amrex::Finalize();
}
