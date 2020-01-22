#ifndef NSIMD_MODULES_SPMD
#define NSIMD_MODULES_SPMD

#include <nsimd/nsimd-all.hpp>

// #############################################################################
// #############################################################################
// #############################################################################

#if NSIMD_CXX > 0 && NSIMD_CXX < 2011 && defined(NSIMD_IS_NVCC)
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

  template <typename U> explicit cuda_allocator(cuda_allocator<U> const &) {}

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
    cudaError_t r = cudaMemcpy(p, &t, sizeof(T),
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
// #############################################################################
// #############################################################################

namespace spmd {

struct scalar_t {};
struct simd_t {};
struct cuda_t {};
struct hip_t {};

#ifdef NSIMD_IS_NVCC

// ----------------------------------------------------------------------------
// CUDA version

#define ALLOC nsimd::cuda_allocator

#define KERNEL(name, ...)                                                     \
  __global__ void name(Device const device, __VA_ARGS__)

#define GET_THREAD_ID() blockIdx.x

// clang-format off
#define CALL_ELTWISE(T, name, size, ...) \
  name<<<size, 1>>>(spmd::cuda_t(), __VA_ARGS__);
// clang-format on

#define SET(src) src
#define LOAD(src) src
#define STORE(dst, src) dst = src

#define IF(cond) if (cond) {
#define ENDIF() }

#else

// ----------------------------------------------------------------------------
// SIMD version

// Kernel

#define ALLOC nsimd::allocator

struct with_mask_t {};
struct without_mask_t {};
struct void_t {};

#define KERNEL(name, ...)                                                     \
  inline void name(Device const device, nat thread_id_,                       \
                   spmd::without_mask_t const mask_or_not,                    \
                   spmd::void_t const pack_logical, __VA_ARGS__)

#define GET_THREAD_ID() thread_id_

#define CALL_ELTWISE(T, name, size, ...)                                      \
  {                                                                           \
    nat i;                                                                    \
    nat len = nsimd::len(nsimd::pack<T>());                                   \
    /*__asm__ __volatile__("cpuid");*/                                        \
    for (i = 0; i + len - 1 < size; i += len) {                               \
      name(spmd::simd_t(), i, spmd::without_mask_t(), spmd::void_t(),         \
           __VA_ARGS__);                                                      \
    }                                                                         \
    /*__asm__ __volatile__("cpuid");*/                                        \
    for (; i < size; i++) {                                                   \
      name(spmd::scalar_t(), i, spmd::without_mask_t(), spmd::void_t(),       \
           __VA_ARGS__);                                                      \
    }                                                                         \
  }

// Set

template <typename T> T set(spmd::scalar_t const /*device*/, T const src) {
  return src;
}

template <typename T>
nsimd::pack<T> set(spmd::simd_t const /*device*/, T const src) {
  return nsimd::pack<T>(src);
}

#define SET(src) spmd::set(device, src)

// Load

template <typename T> T load(spmd::scalar_t const /*device*/, T *const src) {
  return *src;
}

template <typename T>
nsimd::pack<T> load(spmd::simd_t const /*device*/, T *const src) {
  return nsimd::loadu<nsimd::pack<T> >(src);
}

#define LOAD(src) spmd::load(device, &(src))

// Store

template <typename T, typename WithMaskOrNot, typename PackLogical>
void store(spmd::scalar_t const /*device*/, T *const dst, T const src,
           WithMaskOrNot const /*with_mask_or_not*/,
           PackLogical const /*pack_logical*/) {
  *dst = src;
}

template <typename T>
void store(spmd::simd_t const /*device*/, T *const dst,
           nsimd::pack<T> const src,
           spmd::without_mask_t const /*without_mask*/,
           spmd::void_t const /*pack_logical*/) {
  nsimd::storeu(dst, src);
}

template <typename T, typename PackLogical>
void store(spmd::simd_t const /*device*/, T *const dst,
           nsimd::pack<T> const src, spmd::with_mask_t const /*with_mask*/,
           PackLogical const pack_logical) {
  nsimd::storeu(dst, nsimd::if_else1(pack_logical, src,
                                     nsimd::loadu<nsimd::pack<T> >(dst)));
}

#define STORE(dst, src)                                                       \
  spmd::store(device, &(dst), src, mask_or_not, pack_logical)

// If

#define IF(cond)                                                              \
  {                                                                           \
    spmd::with_mask_t mask_or_not;                                            \
    nsimd::packl<int> pack_logical = cond;

#define ENDIF() }

#endif

} // namespace spmd

#endif
