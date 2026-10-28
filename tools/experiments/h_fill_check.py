import os
# Author 14:42: when a gated block's H bit is missing (late or skipped), what to fill it with for the int8 kernel.
# Q8_Z values on the real blk.20 attn_qkv (rotated), u = q + 127 = 2b + h; base-only decodes compared.
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'q7zh_check.py')).read().split("q7c = kq(201, w)")[0])
Q8, S8 = zip_qs(8); u = Q8 + 127; b = np.floor_divide(u, 2)
cases = {
  'H read (Q8_Z)': Q8,
  'fill h = 0': 2 * b - 127,
  'fill h = 1': 2 * b + 1 - 127,
  'fill alternating 0,1': 2 * b + (np.arange(w.shape[1]) % 2)[None, :] - 127,
  'fill toward zero': np.where(2 * b - 127 < 0, 2 * b + 1 - 127, 2 * b - 127),
  'fill 0.5 (int8 2b-127 + 0.5*s*sum(x) per block)': 2 * b + 0.5 - 127,
}
for n, q in cases.items(): print('  %-48s weighted error %.6f' % (n, errs(q * S8)[1]))
