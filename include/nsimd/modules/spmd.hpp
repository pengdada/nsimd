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
template <typename T, int N, typename SimdExt> struct to_logical_helper_t {
  typedef nsimd::packl<T, N, SimdExt> type;
};
template <typename T> to_logical_t = typename to_logical_helper_t<T>::type;

#define KERNEL(name, ...)                                                     \
  template <typename FLOAT>                                                   \
  inline name(nat thread_id_, to_logical_t<FLOAT> thread_mask_, __VA_ARGS__)

#define GET_THREADID() thread_id_

#define CALL_ELTWISE(name, size, ...)                                         \
  {                                                                           \
    nat i;                                                                    \
    for (i = 0; i < size; i += nsimd::len<nsimd::pack<float> >()) {           \
      name<nsimd::pack<float> >(i, __VA_ARGS__);                              \
    }                                                                         \
  }

#endif

} // namespace nsimd

#endif
