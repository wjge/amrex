#ifndef AMREX_MF_PARALLEL_FOR_G_H_
#define AMREX_MF_PARALLEL_FOR_G_H_
#include <AMReX_Config.H>

#ifdef AMREX_USE_GPU

#include <algorithm>
#include <cmath>
#include <limits>

namespace amrex {
namespace detail {

inline
std::pair<int*,int*> build_par_for_nblocks (Vector<Long> const& ncells)
{
    int* hp = nullptr;
    int* dp = nullptr;
    if (!ncells.empty()) {
        const int nboxes = ncells.size();
        const std::size_t nbytes = (nboxes+1) * sizeof(int);
        hp = (int*)The_Pinned_Arena()->alloc(nbytes);
        hp[0] = 0;
        Long ntot = 0;
        bool same_size = true;
        for (int i = 0; i < nboxes; ++i) {
            Long nblocks = (ncells[i] + AMREX_GPU_MAX_THREADS-1) / AMREX_GPU_MAX_THREADS;
            hp[i+1] = hp[i] + static_cast<int>(nblocks);
            ntot += nblocks;
            same_size = same_size && (ncells[i] == ncells[0]);
        }
        amrex::ignore_unused(ntot);
        AMREX_ASSERT(static_cast<Long>(hp[nboxes]) == ntot); // no overflow
        if (!same_size) {
            dp = (int*) The_Arena()->alloc(nbytes);
            Gpu::htod_memcpy(dp, hp, nbytes);
        }
    }
    return std::make_pair(hp,dp);
}

inline
void destroy_par_for_nblocks (std::pair<int*,int*> const& pp)
{
    The_Pinned_Arena()->free(pp.first);
    The_Arena()->free(pp.second);
}
}

namespace experimental {
namespace detail {

namespace parfor_mf_detail {
    template <typename F>
    AMREX_GPU_DEVICE
    auto call_f (F const& f, int b, int i, int j, int k, int) noexcept
        -> decltype(f(0,0,0,0))
    {
        f(b,i,j,k);
    }

    template <typename F>
    AMREX_GPU_DEVICE
    auto call_f (F const& f, int b, int i, int j, int k, int n) noexcept
        -> decltype(f(0,0,0,0,0))
    {
        f(b,i,j,k,n);
    }
}

template <typename MF, typename F>
std::enable_if_t<IsFabArray<MF>::value>
ParallelFor (MF const& mf, IntVect const& nghost, int ncomp, IntVect const&, F&& f)
{
    const auto& index_array = mf.IndexArray();
    const int nboxes = index_array.size();
    if (nboxes == 0) return;

    AMREX_ASSERT(nghost.allLE(mf.nGrowVect()) && nghost.allGE(IntVect(0)));
    IntVect ngrow = nghost - mf.nGrowVect(); // We use this to go from fabbox to valid+nghost.
    auto ma = mf.arrays();

    auto par_for_blocks = mf.getParForInfo(nghost).getBlocks();
    const int nblocks = par_for_blocks.first[nboxes];
    const int block_0_size = par_for_blocks.first[1];
    const int* dp_nblocks = par_for_blocks.second;

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)

    amrex::launch_global<AMREX_GPU_MAX_THREADS>
        <<<nblocks, AMREX_GPU_MAX_THREADS, 0, Gpu::gpuStream()>>>
        ([=] AMREX_GPU_DEVICE () noexcept
         {
             int ibox, icell;
             if (dp_nblocks) {
                 ibox = amrex::bisect(dp_nblocks, 0, nboxes, static_cast<int>(blockIdx.x));
                 icell = (blockIdx.x-dp_nblocks[ibox])*AMREX_GPU_MAX_THREADS + threadIdx.x;
             } else {
                 ibox = blockIdx.x / block_0_size;
                 icell = (blockIdx.x-ibox*block_0_size)*AMREX_GPU_MAX_THREADS + threadIdx.x;
             }

#elif defined(AMREX_USE_DPCPP)

    amrex::launch(nblocks, AMREX_GPU_MAX_THREADS, Gpu::gpuStream(),
         [=] AMREX_GPU_DEVICE (sycl::nd_item<1> const& item) noexcept
         {
             int ibox, icell;
             int blockIdxx = item.get_group_linear_id();
             int threadIdxx = item.get_local_linear_id();
             if (dp_nblocks) {
                 ibox = amrex::bisect(dp_nblocks, 0, nboxes, static_cast<int>(blockIdxx));
                 icell = (blockIdxx-dp_nblocks[ibox])*AMREX_GPU_MAX_THREADS + threadIdxx;
             } else {
                 ibox = blockIdxx / block_0_size;
                 icell = (blockIdxx-ibox*block_0_size)*AMREX_GPU_MAX_THREADS + threadIdxx;
             }
#endif
             Box b(ma[ibox]);
             b.grow(ngrow);
             int ncells = b.numPts();
             if (icell < ncells) {
                 const auto len = amrex::length(b);
                 int k =  icell /   (len.x*len.y);
                 int j = (icell - k*(len.x*len.y)) /   len.x;
                 int i = (icell - k*(len.x*len.y)) - j*len.x;
                 AMREX_D_TERM(i += b.smallEnd(0);,
                              j += b.smallEnd(1);,
                              k += b.smallEnd(2);)
                 for (int n = 0; n < ncomp; ++n) {
                     parfor_mf_detail::call_f(f, ibox, i, j, k, n);
                 }
             }
         });

    AMREX_GPU_ERROR_CHECK();
}

template <typename MF, typename F>
std::enable_if_t<IsFabArray<MF>::value>
ParallelFor (MF const& mf, IntVect const& nghost, IntVect const& ts, F&& f)
{
    ParallelFor(mf, nghost, 1, ts, std::forward<F>(f));
}

}
}
}

#endif
#endif
