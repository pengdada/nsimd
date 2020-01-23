/*

Copyright (c) 2019 Agenium Scale

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include <nsimd/nsimd-all.hpp>

template <typename T, int N> int test_TN() {
  T buf[unsigned(N * nsimd::max_len_t<T>::value)];

  // store true
  nsimd::storelu(buf, nsimd::packl<T, N>(true));
  for (int i = 0; i < nsimd::len(nsimd::pack<T, N>()); i++) {
    if (buf[i] == T(0)) {
      return -1;
    }
  }

  // store false
  nsimd::storelu(buf, nsimd::packl<T, N>(false));
  for (int i = 0; i < nsimd::len(nsimd::pack<T, N>()); i++) {
    if (buf[i] != T(0)) {
      return -1;
    }
  }

  return 0;
}

template <int N> int test_N() {
  return test_TN<i8, N>() || test_TN<u8, N>() || test_TN<i16, N>() ||
         test_TN<u16, N>() || test_TN<i32, N>() || test_TN<u32, N>() ||
         test_TN<i64, N>() || test_TN<u64, N>() || test_TN<f32, N>() ||
         test_TN<f64, N>();
}

int main() { return test_N<1>() || test_N<2>() || test_N<3>() || test_N<4>(); }
