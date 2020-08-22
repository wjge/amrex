#include <AMReX.H>
#include <AMReX_Gpu.H>

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    //warm up
    for(int j;j<100;j++)
    {
        AMREX_FOR_1D (460000000,i,{double dummy0=1.0; double dummy1=dummy0+32.232*dummy0;});
    }

    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<float> fsec;

    Gpu::Device::synchronize();
    auto start_clock = Time::now();

    AMREX_FOR_1D (460000000,i,{double dummy0=1.0; double dummy1=dummy0+32.232*dummy0;});

    Gpu::Device::synchronize();
    auto finish_clock = Time::now();
    fsec fs = finish_clock - start_clock;
    std::cout<<"time taken for amrex_parallel_for (msecs):" << fs.count()*1e3 << std::endl;

    amrex::Finalize();
}
