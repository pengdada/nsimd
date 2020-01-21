#ifndef NSIMD_MODULES_SPMD
#define NSIMD_MODULES_SPMD

#include <nsimd/nsimd-all.hpp>

// #############################################################################

#if NSIMD_CXX > 0 && NSIMD_CXX < 2011 && defined(NSIMD_IS_NVCC)
#include <iostream>
#include <stdexcept>
          namespace nsimd {

    template <typename T> class cuda_allocator {
    public:
      typedef T value_type;
      typedef value_type *pointer;
      typedef const value_type *const_pointer;
      typedef value_type &reference;
      typedef const value_type &const_reference;
      typedef std::size_t size_type;
      typedef std::ptrdiff_t difference_type;

    public:
      template <typename U> struct rebind { typedef cuda_allocator<U> other; };

    public:
      cuda_allocator() {}
      ~cuda_allocator() {}
      cuda_allocator(cuda_allocator const &) {}

      template <typename U>
      explicit cuda_allocator(cuda_allocator<U> const &) {}

      pointer address(reference r) { return &r; }
      const_pointer address(const_reference r) { return &r; }

      pointer allocate(size_type n) {
        pointer p = NULL;
        if (cudaMalloc(&p, n * sizeof(T)) != cudaSuccess) {
          throw std::bad_alloc();
        }
        return p;
      }

      pointer allocate(size_type n, const void *) { return allocate(n); }

      void deallocate(pointer p, size_type) {
        cudaError_t r = cudaFree(p);
        if (r != cudaSuccess) {
          if (r == cudaErrorInvalidDevicePointer) {
            throw std::domain_error("cudaFree: Invalid device pointer");
          } else if (r == cudaErrorInitializationError) {
            throw std::runtime_error("cudaFree: Initialization error");
          } else {
            throw std::runtime_error("cudaFree fails");
          }
        }
      }

      size_type max_size() const { return size_type(-1) / sizeof(T); }

      void construct(pointer p, const T &t) {
        cudaError_t r =
            cudaMemcpy(p, &t, sizeof(T),
                       /*cudaMemcpyHostToDevice*/ cudaMemcpyDefault);
        if (r != cudaSuccess) {
          if (r == cudaErrorInvalidValue) {
            throw std::domain_error("cudaMemcpy: Invalid value");
          } else if (r == cudaErrorInvalidMemcpyDirection) {
            throw std::runtime_error("cudaMemcpy: Invalid memcpy direction");
          } else if (r == cudaErrorInitializationError) {
            throw std::runtime_error("cudaMemcpy: Initialization error");
          } else if (r == cudaErrorInsufficientDriver) {
            throw std::runtime_error("cudaMemcpy: Insufficient driver error");
          } else if (r == cudaErrorNoDevice) {
            throw std::runtime_error("cudaMemcpy: No device");
          } else if (r == cudaErrorNotPermitted) {
            throw std::runtime_error("cudaMemcpy: Not permitted");
          } else {
            throw std::runtime_error("cudaMemcpy fails");
          }
        }
        // cudaDeviceSynchronize();
      }
      void destroy(pointer /*p*/) {}

      bool operator==(cuda_allocator const &) { return true; }
      bool operator!=(cuda_allocator const &a) { return !operator==(a); }
    };

  } // namespace nsimd
#endif

#if NSIMD_CXX >= 2011 && defined(NSIMD_IS_NVCC)
namespace nsimd {

template <typename T> struct cuda_allocator {
  using value_type = T;

  cuda_allocator() = default;

  template <typename S> cuda_allocator(cuda_allocator<S> const &) {}

  T *allocate(std::size_t n) {
    T *p = NULL;
    if (cudaMalloc(&p, n) != cudaSuccess) {
      throw std::bad_alloc();
    }
    return p;
  }

  void deallocate(T *p, std::size_t) {
    cudaError_t r = cudaFree(p);
    if (r != cudaSuccess) {
      if (r == cudaErrorInvalidDevicePointer) {
        throw std::domain_error("cudaFree: Invalid device pointer");
      } else if (r == cudaErrorInitializationError) {
        throw std::runtime_error("cudaFree: Initialization error");
      } else {
        throw std::runtime_error("cudaFree fails");
      }
    }
  }
};

} // namespace nsimd
#endif

// #############################################################################

namespace nsimd {

#ifdef NSIMD_IS_NVCC

// ----------------------------------------------------------------------------
// CUDA version

#define ALLOC nsimd::cuda_allocator

#define KERNEL(name, ...) __global__ void name(__VA_ARGS__)

#define GET_THREAD_ID() blockIdx.x

// clang-format off
#define CALL_ELTWISE(T, name, size, ...) name<<<size, 1>>>(__VA_ARGS__);
// clang-format on

#define STORE(dst, src) dst = src
#define LOAD(T, src) src

#define IF(cond) if (cond) {
#define ENDIF() }

#else

// ----------------------------------------------------------------------------
// CPU version

#define ALLOC nsimd::allocator

struct do_it_t { };
struct do_not_it_t { };

template <typename T> struct to_logical_helper_t { typedef bool type; };
template <typename T, int N, typename SimdExt>
struct to_logical_helper_t<nsimd::pack<T, N, SimdExt> > {
  typedef nsimd::packl<T, N, SimdExt> type;
};
template <typename T>
using to_logical_t = typename to_logical_helper_t<T>::type;

struct no_mask_t {};
struct with_mask_t {};

// #define KERNEL(name, ...)                                                     \
//   template <typename T>                                                       \
//   inline void name(nat thread_id_, nsimd::to_logical_t<T> thread_mask_,       \
//                    nsimd::no_mask_t need_mask, __VA_ARGS__)

// #define KERNEL(name, ...)                                                     \
// <<<<<<< HEAD
//   template <typename FLOAT>                                                   \
//   inline void name(int thread_id_, nsimd::no_mask_t need_mask_,               \
//                    nsimd::to_logical_t<FLOAT> thread_mask_, __VA_ARGS__)

#define KERNEL(name, ...)                                                     \
  inline void name(nat thread_id_, nsimd::do_not_it_t do_it_or_do_not_it, __VA_ARGS__)

#define GET_THREAD_ID() thread_id_

/*#define CALL_ELTWISE(T, name, size, ...)                                      \
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
#define LOAD(src) nsimd::loadu<nsimd::pack<float> >(&(src))*/

#define CALL_ELTWISE(T, name, size, ...)                                      \
  {                                                                           \
    nat i;                                                                    \
    /*__asm__ __volatile__("cpuid");*/                                        \
    nat len = nsimd::len(nsimd::pack<T>());                                   \
    for (i = 0; i + len - 1 < size; i += len) {                               \
      name /*<nsimd::pack<T> >*/ (                                            \
          i, nsimd::do_not_it_t() /*, nsimd::packl<T>(true), nsimd::no_mask_t()*/, __VA_ARGS__);    \
    }                                                                         \
    /*__asm__ __volatile__("cpuid");*/                                        \
    for (; i < size; i++) {                                                   \
      name /*<T>*/ (i, nsimd::do_not_it_t() /*, true, nsimd::no_mask_t()*/, __VA_ARGS__);           \
    }                                                                         \
  }

template <typename T0, typename T1>
void store_or_not(T0 t0, T1 t1, nsimd::do_it_t const) {
  nsimd::storeu(t0, t1);
}

template <typename T0, typename T1>
void store_or_not(T0 /*t0*/, T1 /*t1*/, nsimd::do_not_it_t const) {
  // Do nothing
}

// #define STORE(dst, src) nsimd::storeu(&(dst), src)
#define STORE(dst, src) std::cout << "TYPE NAME = " << typeid(do_it_or_do_not_it).name() << std::endl; nsimd::store_or_not(&(dst), src, do_it_or_do_not_it)
#define LOAD(T, src) nsimd::loadu<nsimd::pack<T> >(&(src))

#define IF(cond) { nsimd::do_it_t do_it_or_do_not_it;
#define ENDIF() }

// #define IF(cond)                                                              \
//   nsimd::to_logical_t<FLOAT> thread_new_mask_ = (cond);                       \
//   auto thread_then_code_ = [&](nsimd::to_logical_t<FLOAT> thread_mask_) {
//
// #define ELSE                                                                  \
//   }                                                                           \
//   ;                                                                           \
//   auto thread_else_code_ = [&](nsimd::to_logical_t<FLOAT> thread_mask_) {
//
// #define ENDIF                                                                 \
//   }                                                                           \
//   ;

#endif

} // namespace nsimd

#endif
