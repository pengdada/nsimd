#include <nsimd/modules/spmd.hpp>

KERNEL(test_add, float *dst, float *src1, float *src2) {
  nat i = GET_THREAD_ID();
  STORE(dst[i], LOAD(src1[i]) + LOAD(src2[i]));
}

int main(void) {
  const int N = 1000;
  float a[N], b[N], res[N];
  for (int i = 0; i < N; i++) {
    a[i] = float(i);
    b[i] = float(i);
  }

  CALL_ELTWISE(test_add, N, res, a, b);

  for (int i = 0; i < N; i++) {
    if (res[i] != a[i] + b[i]) {
      return -1;
    }
  }
  return 0;
}
