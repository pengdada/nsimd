#include <nsimd/modules/spmd.hpp>

KERNEL(test_add, float *dst, float *src1, float *src2) {
  nat i = GET_THREADID();
  STORE(dst[i], LOAD(src1[i]) + LOAD(src2[i]));
}

int main(void) {
  float a[1000], b[1000], c[1000];
  CALL_ELTWISE(test_add, 1000, a, b, c);
  return 0;
}
