# The multiply-accumulate unit's arithmetic (ROADMAP, Stage 1), the model its hardware must match bit for bit: acc + a * b,
# a and b bf16, acc and the result fp32, each a bit pattern. The product is exact unless it is out of fp32's range (then
# infinity, or zero); the sum is rounded to nearest, ties to even; a denormal in or out is flushed to zero, keeping its
# sign; inf x 0 and inf - inf give NaN, and every NaN out is 0x7FC00000. Integers throughout, so the model leans on no
# float hardware's behavior: each value is (-1)^sign * m * 2^e.
NAN, INF = 0x7FC00000, 0x7F800000

def fields(bits, mbits):  # sign, biased exponent, mantissa (8 exponent bits, as bf16 and fp32 both have)
  return bits >> (mbits + 8) & 1, bits >> mbits & 255, bits & ((1 << mbits) - 1)

def mac(acc, a, b):
  (sa, ea, ma), (sb, eb, mb), (sc, ec, mc) = fields(a, 7), fields(b, 7), fields(acc, 23)
  if (ea == 255 and ma) or (eb == 255 and mb) or (ec == 255 and mc): return NAN
  sp, zero, inf = sa ^ sb, ea == 0 or eb == 0, ea == 255 or eb == 255  # the product's sign; a zero (or denormal) factor, an infinite one
  if inf and zero: return NAN
  mp, ep = ((128 | ma) * (128 | mb), ea + eb - 268) if not (inf or zero) else (0, 0)  # exact: 16 bits of significand
  top = mp.bit_length() - 1 + ep  # the product's exponent, unbiased
  if inf or mp and top >= 128: return NAN if ec == 255 and sc != sp else sp << 31 | INF  # infinite, or too big for fp32
  if ec == 255: return acc  # infinity plus a finite product
  if mp and top < -126: mp = 0  # too small for fp32: flushed
  mc, ec = (0, 0) if ec == 0 else (1 << 23 | mc, ec - 150)  # acc's significand and exponent; a denormal flushed
  if mp == 0 and mc == 0: return (sp & sc) << 31  # -0 only from -0 + -0
  e = min(ep, ec) if mp and mc else ep if mp else ec  # the exact sum, as an integer times 2^e
  total = (-mp if sp else mp) * (1 << ep - e if mp else 0) + (-mc if sc else mc) * (1 << ec - e if mc else 0)
  if total == 0: return 0  # cancelled: +0, as round to nearest gives
  sign, m = int(total < 0), abs(total)
  shift = m.bit_length() - 24  # to 24 bits of significand, rounded to nearest, ties to even
  if shift > 0:
    q, r, half = m >> shift, m & ((1 << shift) - 1), 1 << (shift - 1)
    m, e = q + (r > half or r == half and q & 1), e + shift
    if m >> 24: m, e = m >> 1, e + 1
  else: m, e = m << -shift, e + shift
  biased = e + 150
  if biased >= 255: return sign << 31 | INF
  if biased <= 0: return sign << 31  # too small for a normal fp32: flushed (exact here, so before or after rounding agree)
  return sign << 31 | biased << 23 | m & 0x7FFFFF
