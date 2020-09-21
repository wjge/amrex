#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <cuda_profiler_api.h>

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    const double dt = 0.001;
    const int N = 5000000;
{
    Gpu::DeviceVector<Real> x_d(N);
    Gpu::DeviceVector<Real> y_d(N);

    auto x_d_ptr = x_d.dataPtr();
    auto y_d_ptr = y_d.dataPtr();

    //warm up
    for(int j=0;j<100;j++)
    {
        AMREX_FOR_1D (N,i,{x_d_ptr[i]=x_d_ptr[i]+y_d_ptr[i]*dt;});
    }

    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<float> fsec;

    int Nl=1000;

    Gpu::Device::synchronize();
    auto start_clock = Time::now();

    cudaProfilerStart();

    for(int j=0;j<Nl;j++)
    {
        AMREX_FOR_1D (N,i,{x_d_ptr[i]=x_d_ptr[i]+y_d_ptr[i]*dt;});
    }

    Gpu::Device::synchronize();

    cudaProfilerStop();

    auto finish_clock = Time::now();
    fsec fs = finish_clock - start_clock;
    std::cout<<"time taken for amrex_parallel_for (msecs):" << fs.count()*1e3/Nl << std::endl;
}
    amrex::Finalize();
}
