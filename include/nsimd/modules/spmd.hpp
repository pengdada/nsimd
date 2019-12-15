#ifndef NSIMD_MODULES_SPMD
#define NSIMD_MODULES_SPMD

#include <nsimd/nsimd-all.hpp>

namespace nsimd {

// ----------------------------------------------------------------------------
// CUDA version

#ifdef NSIMD_IS_NVCC

// TODO

// ----------------------------------------------------------------------------
// CPU version

#else

template <typename T> struct to_logical_helper_t { typedef bool type; };
template <typename T, int N, typename SimdExt>
struct to_logical_helper_t<nsimd::pack<T, N, SimdExt> > {
  typedef nsimd::packl<T, N, SimdExt> type;
};
template <typename T>
using to_logical_t = typename to_logical_helper_t<T>::type;

struct no_mask_t {};
struct with_mask_t {};

#define KERNEL(name, ...)                                                     \
  template <typename FLOAT>                                                   \
  inline void name(nat thread_id_, nsimd::to_logical_t<FLOAT> thread_mask_,   \
                   nsimd::no_mask_t need_mask, __VA_ARGS__)

#define GET_THREAD_ID() thread_id_

#define CALL_ELTWISE(name, size, ...)                                         \
  {                                                                           \
    nat i;                                                                    \
    __asm__ __volatile__("cpuid");                                            \
    __asm__ __volatile__("cpuid");                                            \
    __asm__ __volatile__("cpuid");                                            \
    __asm__ __volatile__("cpuid");                                            \
    for (i = 0; i < size; i += nsimd::len(nsimd::pack<float>())) {            \
      name<nsimd::pack<float> >(i, nsimd::packl<float>(true),                 \
                                nsimd::no_mask_t(), __VA_ARGS__);             \
    }                                                                         \
    __asm__ __volatile__("cpuid");                                            \
    __asm__ __volatile__("cpuid");                                            \
    __asm__ __volatile__("cpuid");                                            \
    __asm__ __volatile__("cpuid");                                            \
    for (; i < size; i++) {                                                   \
      name<float>(i, true, nsimd::no_mask_t(), __VA_ARGS__);                  \
    }                                                                         \
  }

#define STORE(dst, src) nsimd::storeu(&(dst), src)
#define LOAD(src) nsimd::loadu<nsimd::pack<float> >(&(src))

#endif

} // namespace nsimd

#endif
