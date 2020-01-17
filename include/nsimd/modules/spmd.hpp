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
  inline void name(int thread_id_, nsimd::no_mask_t need_mask_,               \
                   nsimd::to_logical_t<FLOAT> thread_mask_, __VA_ARGS__)

#define GET_THREAD_ID() thread_id_

#define CALL_ELTWISE(name, size, ...)                                         \
  {                                                                           \
    int thread_id_;                                                           \
    nsimd::packl<float> thread_all_true_(true);                               \
    int len = nsimd::len(nsimd::pack<float>());                               \
    for (thread_id_ = 0; thread_id_ < size; thread_id_ += len) {              \
      name<nsimd::pack<float> >(thread_id_, nsimd::no_mask_t(),               \
                                thread_all_true_, __VA_ARGS__);               \
    }                                                                         \
    for (; thread_id_ < size; thread_id_++) {                                 \
      name<float>(thread_id_, nsimd::no_mask_t(), true, __VA_ARGS__);         \
    }                                                                         \
  }

template <typename NeedMask, typename Type> struct store_mov_helper_t {};

template <> struct store_mov_helper_t<nsimd::no_mask_t, float> {
  static void store(float *dst, Type src, bool) {
    nsimd::storeu(dst, src);
  }
};

template <typename Type> struct store_mov_helper_t<nsimd::with_mask_t, Type> {
  static void store(float *dst, Type src) {
    Type
    nsimd::storeu(dst, src);
  }
};

#define STORE(dst, src) nsimd::storeu(&(dst), src)
#define LOAD(src) nsimd::loadu<nsimd::pack<float> >(&(src))

#define IF(cond)                                                              \
  nsimd::to_logical_t<FLOAT> thread_new_mask_ = (cond);                       \
  auto thread_then_code_ = [&](nsimd::to_logical_t<FLOAT> thread_mask_) {

#define ELSE                                                                  \
  }                                                                           \
  ;                                                                           \
  auto thread_else_code_ = [&](nsimd::to_logical_t<FLOAT> thread_mask_) {

#define ENDIF                                                                 \
  }                                                                           \
  ;

#endif

} // namespace nsimd

#endif
