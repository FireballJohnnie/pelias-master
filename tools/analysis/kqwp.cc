// provenance: original (sandbox helper). kqw through pipes, no files: kqwp TYPE_ID ROWS NCOLS; stdin = ROWS x NCOLS f32 then
// NCOLS f32 importance; stdout = the dequantized rows (f32). Stderr: bits per weight.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "quant/block_types.h"
#include "quant/quant_cpu.h"
using namespace pelias::quant;
int main(int argc, char** argv) {
  if (argc < 4) return 2;
  const auto t = static_cast<Type>(std::atoi(argv[1]));
  const long rows = std::atol(argv[2]), n = std::atol(argv[3]);
  std::vector<float> x(static_cast<size_t>(rows * n)), y(x.size()), imp(static_cast<size_t>(n));
  if (std::fread(x.data(), 4, x.size(), stdin) != x.size()) return 3;
  if (std::fread(imp.data(), 4, imp.size(), stdin) != imp.size()) return 6;
  const auto rb = row_bytes(static_cast<std::uint32_t>(t), static_cast<std::uint64_t>(n));
  std::vector<std::uint8_t> q(rb * static_cast<size_t>(rows));
  if (!quantize_rows(t, x.data(), q.data(), rows, n, imp.data())) return 4;
  for (long r = 0; r < rows; ++r)
    if (!dequantize_row(t, q.data() + rb * r, y.data() + r * n, n)) return 5;
  if (std::fwrite(y.data(), 4, y.size(), stdout) != y.size()) return 7;
  std::fprintf(stderr, "%.4f\n", 8.0 * rb / n);
  return 0;
}
