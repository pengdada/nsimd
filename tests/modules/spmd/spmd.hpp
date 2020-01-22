#include <nsimd/modules/spmd.hpp>

#include <iostream>
#include <vector>

#ifdef NSIMD_IS_NVCC
template <typename Device, typename T>
KERNEL(init, T *v) {
  nat i = GET_THREAD_ID();
  STORE(v[i], int(i));
}
#endif

template <typename Device, typename T>
KERNEL(test_add, T *dst, T *src1, T *src2) {
  nat i = GET_THREAD_ID();
  STORE(dst[i], LOAD(src1[i]) + LOAD(src2[i]));
  IF (LOAD(src1[i]) >= SET(5))
    STORE(dst[i], LOAD(dst[i]) + 100);
  ENDIF()
}

#ifdef NSIMD_IS_NVCC
template <typename Device, typename T>
KERNEL(print, T *dst, T *src1, T *src2) {
  nat i = GET_THREAD_ID();
  printf("%i: %i + %i = %i\n", int(i), src1[i], src2[i], dst[i]);
}
#endif

int main() {
  int const N = 2 * 8 + 3;

  std::vector<int, ALLOC<int> > a(N);
  std::vector<int, ALLOC<int> > b(N);
  std::vector<int, ALLOC<int> > r(N);

  #ifdef NSIMD_IS_NVCC
  CALL_ELTWISE(int, init, N, a.data());
  CALL_ELTWISE(int, init, N, b.data());
  #else
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i;
  }
  #endif

  CALL_ELTWISE(int, test_add, N, r.data(), a.data(), b.data());

  #ifdef NSIMD_IS_NVCC
  CALL_ELTWISE(int, print, N, r.data(), a.data(), b.data());
  #else
  for (int i = 0; i < N; i++) {
    std::cout << i << ": " << a[i] << " + " << b[i] << " = " << r[i] << std::endl;
  }
  #endif

  // for (int i = 0; i < N; i++) {
  //   if (res[i] != a[i] + b[i]) {
  //     return -1;
  //   }
  // }
  return 0;
}
