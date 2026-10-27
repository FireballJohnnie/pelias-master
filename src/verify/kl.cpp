// provenance: original (the base-file layout and every statistic READ-SRC from llama.cpp tools/perplexity/perplexity.cpp
//   (kl_divergence, log_softmax, process_logits) at commit 2ca15f5404760548c39e7b92bd43116a09414a1a, read as a
//   specification; no code copied)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "verify/kl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>

namespace pelias::verify {

Status KlBaseFile::open(const std::string& path) {
  in_.open(path, std::ios::binary);
  if (!in_) return Status::fail(Code::not_found, "the KL-divergence base file could not be opened");
  char magic[8] = {};
  in_.read(magic, 8);
  if (!in_ || std::memcmp(magic, "_logits_", 8) != 0)
    return Status::fail(Code::corrupt, "the file is not a KL-divergence base file (no _logits_ header)", 0);
  in_.read(reinterpret_cast<char*>(&n_ctx_), 4);
  in_.read(reinterpret_cast<char*>(&n_vocab_), 4);
  in_.read(reinterpret_cast<char*>(&n_chunk_), 4);
  if (!in_) return Status::fail(Code::corrupt, "the base file's header is cut short", 8);
  if (n_ctx_ < 4 || n_ctx_ > (1u << 20) || n_vocab_ <= 0 || n_chunk_ <= 0)
    return Status::fail(Code::corrupt, "the base file's header holds an impossible context, vocabulary or chunk count", 8);
  tokens_.resize(static_cast<std::size_t>(n_ctx_) * static_cast<std::size_t>(n_chunk_));
  in_.read(reinterpret_cast<char*>(tokens_.data()), static_cast<std::streamsize>(tokens_.size() * 4));
  if (!in_) return Status::fail(Code::corrupt, "the base file's token list is cut short", 20);
  return Status::good();
}

Status KlBaseFile::read_chunk(std::vector<std::uint16_t>& out) {
  out.resize(static_cast<std::size_t>(scored()) * static_cast<std::size_t>(record_u16()));
  in_.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * 2));
  if (!in_) return Status::fail(Code::corrupt, "the base file ends before the chunk's log-probabilities");
  return Status::good();
}

KlToken kl_token(std::int32_t n_vocab, const float* logits, const std::uint16_t* base, std::int32_t tok) {
  KlToken r;
  float max_logit = logits[0];
  std::int32_t imax = 0;
  for (std::int32_t i = 1; i < n_vocab; ++i)
    if (logits[i] > max_logit) {
      max_logit = logits[i];
      imax = i;
    }
  double sum_exp = 0.0;
  for (std::int32_t i = 0; i < n_vocab; ++i) sum_exp += static_cast<double>(std::exp(logits[i] - max_logit));
  const float log_sum_exp = static_cast<float>(std::log(sum_exp));
  float scale = 0, min_log_prob = 0;
  std::memcpy(&scale, base, 4);
  std::memcpy(&min_log_prob, base + 2, 4);
  const std::uint16_t* lp = base + 4;
  r.nll = max_logit + log_sum_exp - logits[tok];
  r.nll_base = -(scale * static_cast<float>(lp[tok]) + min_log_prob);
  max_logit += log_sum_exp;
  double sum = 0.0;
  std::int32_t imax_base = -1;
  float p_log_base_max = 0;
  for (std::int32_t i = 0; i < n_vocab; ++i) {
    const float p_log_base = scale * static_cast<float>(lp[i]) + min_log_prob;
    if (i == 0 || p_log_base > p_log_base_max) {
      p_log_base_max = p_log_base;
      imax_base = i;
    }
    if (p_log_base > -16.f) {
      const float p_base = std::exp(p_log_base);
      sum += static_cast<double>(p_base * (p_log_base - logits[i] + max_logit));
    }
  }
  r.kld = sum;
  r.same_top = imax == imax_base;
  const float p_base = std::exp(-r.nll_base);
  const float p = std::exp(-r.nll);
  r.p_diff = p - p_base;
  return r;
}

void KlStats::add(const KlToken& t) {
  sum_nll += static_cast<double>(t.nll);
  sum_nll2 += static_cast<double>(t.nll * t.nll);
  sum_nll_base += static_cast<double>(t.nll_base);
  sum_nll_base2 += static_cast<double>(t.nll_base * t.nll_base);
  sum_nll_nll_base += static_cast<double>(t.nll * t.nll_base);
  sum_kld += t.kld;
  sum_kld2 += t.kld * t.kld;
  if (t.same_top) ++n_same_top;
  ++count;
  sum_p_diff += static_cast<double>(t.p_diff);
  const double p_diff2 = static_cast<double>(t.p_diff * t.p_diff);
  sum_p_diff2 += p_diff2;
  sum_p_diff4 += p_diff2 * p_diff2;
  max_p_diff = std::max(max_p_diff, std::fabs(t.p_diff));
  kld_values.push_back(static_cast<float>(t.kld));
  p_diff_values.push_back(t.p_diff);
}

namespace {

struct Mu {
  double f, df;
};

Mu mean_and_uncertainty(double sum, double sum2, std::uint64_t count) {
  if (count < 1) return {0.0, 0.0};
  const double c = static_cast<double>(count);
  const double f = sum / c;
  double df = sum2 / c - f * f;
  df = df > 0 && count > 10 ? std::sqrt(df / (c - 1)) : 0.0;
  return {f, df};
}

double covariance(double suma, double sumb, double sumab, std::uint64_t count) {
  if (count < 10) return 0.0;
  const double c = static_cast<double>(count);
  double var = sumab / c - (suma / c) * (sumb / c);
  var /= c - 1;
  return var;
}

float percentile(const std::vector<float>& v, float fraction) {
  if (fraction <= 0) return v.front();
  if (fraction >= 1) return v.back();
  float p = fraction * static_cast<float>(v.size() - 1);
  const std::size_t ip = static_cast<std::size_t>(p);
  p -= static_cast<float>(ip);
  return (1 - p) * v[ip] + p * v[std::min(ip + 1, v.size() - 1)];
}

float median(const std::vector<float>& v) {
  return v.size() % 2 == 0 ? 0.5f * (v[v.size() / 2] + v[v.size() / 2 - 1]) : v[v.size() / 2];
}

std::string fmt(const char* f, double a, double b) {
  char buf[160];
  std::snprintf(buf, sizeof buf, f, a, b);
  return buf;
}

std::string fmt1(const char* f, double a) {
  char buf[160];
  std::snprintf(buf, sizeof buf, f, a);
  return buf;
}

}  // namespace

std::string kl_chunk_header() {
  return "chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              \xce\x94p RMS            Same top p\n";
}

std::string kl_chunk_line(int chunk, const KlStats& s) {
  char head[16];
  std::snprintf(head, sizeof head, "%4d", chunk);
  std::string out = head;
  const Mu log_ppl = mean_and_uncertainty(s.sum_nll, s.sum_nll2, s.count);
  const double ppl_val = std::exp(log_ppl.f);
  out += fmt("    %9.4lf \xc2\xb1 %9.4lf", ppl_val, ppl_val * log_ppl.df);
  const Mu log_base = mean_and_uncertainty(s.sum_nll_base, s.sum_nll_base2, s.count);
  const double cov = covariance(s.sum_nll, s.sum_nll_base, s.sum_nll_nll_base, s.count);
  out += fmt("    %10.5lf \xc2\xb1 %10.5lf", log_ppl.f - log_base.f, std::sqrt(log_ppl.df * log_ppl.df + log_base.df * log_base.df - 2.0 * cov));
  const Mu kl = mean_and_uncertainty(s.sum_kld, s.sum_kld2, s.count);
  out += fmt("    %10.5lf \xc2\xb1 %10.5lf", kl.f, kl.df);
  const Mu mse = mean_and_uncertainty(s.sum_p_diff2, s.sum_p_diff4, s.count);
  const double rms = std::sqrt(mse.f);
  out += fmt("    %6.3lf \xc2\xb1 %6.3lf %%", 100.0 * rms, 100.0 * (0.5 / rms * mse.df));
  const double top = 1.0 * static_cast<double>(s.n_same_top) / static_cast<double>(s.count);
  out += fmt("    %6.3lf \xc2\xb1 %6.3lf %%", 100.0 * top, 100.0 * std::sqrt(top * (1 - top) / static_cast<double>(s.count - 1)));
  return out + "\n";
}

std::string kl_report(const KlStats& s) {
  if (s.count < 100) return "too few scored tokens for the closing statistics\n";
  std::vector<float> kv = s.kld_values, pv = s.p_diff_values;
  std::sort(kv.begin(), kv.end());
  std::sort(pv.begin(), pv.end());
  std::string o = "====== Perplexity statistics ======\n";
  const Mu log_ppl = mean_and_uncertainty(s.sum_nll, s.sum_nll2, s.count);
  const double ppl_val = std::exp(log_ppl.f), ppl_unc = ppl_val * log_ppl.df;
  o += fmt("Mean PPL(Q)                   : %10.6lf \xc2\xb1 %10.6lf\n", ppl_val, ppl_unc);
  const Mu log_base = mean_and_uncertainty(s.sum_nll_base, s.sum_nll_base2, s.count);
  const double base_val = std::exp(log_base.f), base_unc = base_val * log_base.df;
  o += fmt("Mean PPL(base)                : %10.6lf \xc2\xb1 %10.6lf\n", base_val, base_unc);
  const double cov = covariance(s.sum_nll, s.sum_nll_base, s.sum_nll_nll_base, s.count);
  o += fmt1("Cor(ln(PPL(Q)), ln(PPL(base))): %6.2lf%%\n", 100.0 * cov / (log_ppl.df * log_base.df));
  const double ratio_val = log_ppl.f - log_base.f;
  const double ratio_unc = std::sqrt(log_ppl.df * log_ppl.df + log_base.df * log_base.df - 2.0 * cov);
  o += fmt("Mean ln(PPL(Q)/PPL(base))     : %10.6lf \xc2\xb1 %10.6lf\n", ratio_val, ratio_unc);
  o += fmt("Mean PPL(Q)/PPL(base)         : %10.6lf \xc2\xb1 %10.6lf\n", std::exp(ratio_val), std::exp(ratio_val) * ratio_unc);
  const double ppl_cov = ppl_val * base_val * cov;
  o += fmt("Mean PPL(Q)-PPL(base)         : %10.6lf \xc2\xb1 %10.6lf\n", ppl_val - base_val,
           std::sqrt(ppl_unc * ppl_unc + base_unc * base_unc - 2.0 * ppl_cov));
  o += "\n====== KL divergence statistics ======\n";
  const Mu kl = mean_and_uncertainty(s.sum_kld, s.sum_kld2, s.count);
  o += fmt("Mean    KLD: %10.6lf \xc2\xb1 %10.6lf\n", kl.f, kl.df);
  o += fmt1("Maximum KLD: %10.6f\n", static_cast<double>(kv.back()));
  o += fmt1("99.9%%   KLD: %10.6f\n", static_cast<double>(percentile(kv, 0.999f)));
  o += fmt1("99.0%%   KLD: %10.6f\n", static_cast<double>(percentile(kv, 0.990f)));
  o += fmt1("95.0%%   KLD: %10.6f\n", static_cast<double>(percentile(kv, 0.950f)));
  o += fmt1("90.0%%   KLD: %10.6f\n", static_cast<double>(percentile(kv, 0.900f)));
  o += fmt1("Median  KLD: %10.6f\n", static_cast<double>(median(kv)));
  o += fmt1("10.0%%   KLD: %10.6f\n", static_cast<double>(percentile(kv, 0.100f)));
  o += fmt1(" 5.0%%   KLD: %10.6f\n", static_cast<double>(percentile(kv, 0.050f)));
  o += fmt1(" 1.0%%   KLD: %10.6f\n", static_cast<double>(percentile(kv, 0.010f)));
  o += fmt1(" 0.1%%   KLD: %10.6f\n", static_cast<double>(percentile(kv, 0.001f)));
  o += fmt1("Minimum KLD: %10.6f\n", static_cast<double>(kv.front()));
  o += "\n====== Token probability statistics ======\n";
  const Mu pd = mean_and_uncertainty(s.sum_p_diff, s.sum_p_diff2, s.count);
  o += fmt("Mean    \xce\x94p: %6.3lf \xc2\xb1 %5.3lf %%\n", 100.0 * pd.f, 100.0 * pd.df);
  o += fmt1("Maximum \xce\x94p: %6.3lf%%\n", 100.0 * static_cast<double>(pv.back()));
  const std::pair<const char*, float> rows[] = {{"99.9%%   \xce\x94p: %6.3lf%%\n", 0.999f}, {"99.0%%   \xce\x94p: %6.3lf%%\n", 0.990f},
                                                {"95.0%%   \xce\x94p: %6.3lf%%\n", 0.950f}, {"90.0%%   \xce\x94p: %6.3lf%%\n", 0.900f},
                                                {"75.0%%   \xce\x94p: %6.3lf%%\n", 0.750f}};
  for (const auto& r : rows) o += fmt1(r.first, 100.0 * static_cast<double>(percentile(pv, r.second)));
  o += fmt1("Median  \xce\x94p: %6.3lf%%\n", 100.0 * static_cast<double>(median(pv)));
  const std::pair<const char*, float> rows2[] = {{"25.0%%   \xce\x94p: %6.3lf%%\n", 0.250f}, {"10.0%%   \xce\x94p: %6.3lf%%\n", 0.100f},
                                                 {" 5.0%%   \xce\x94p: %6.3lf%%\n", 0.050f}, {" 1.0%%   \xce\x94p: %6.3lf%%\n", 0.010f},
                                                 {" 0.1%%   \xce\x94p: %6.3lf%%\n", 0.001f}};
  for (const auto& r : rows2) o += fmt1(r.first, 100.0 * static_cast<double>(percentile(pv, r.second)));
  o += fmt1("Minimum \xce\x94p: %6.3lf%%\n", 100.0 * static_cast<double>(pv.front()));
  const Mu mse = mean_and_uncertainty(s.sum_p_diff2, s.sum_p_diff4, s.count);
  const double rms = std::sqrt(mse.f);
  o += fmt("RMS \xce\x94p    : %6.3lf \xc2\xb1 %5.3lf %%\n", 100.0 * rms, 100.0 * (0.5 / rms * mse.df));
  const double top = 1.0 * static_cast<double>(s.n_same_top) / static_cast<double>(s.count);
  o += fmt("Same top p: %6.3lf \xc2\xb1 %5.3lf %%\n", 100.0 * top, 100.0 * std::sqrt(top * (1.0 - top) / static_cast<double>(s.count - 1)));
  return o;
}

std::uint64_t kl_token_hash(const std::int32_t* tokens, std::size_t n) {
  std::uint64_t h = 1469598103934665603ull;
  for (std::size_t i = 0; i < n; ++i) {
    const auto u = static_cast<std::uint32_t>(tokens[i]);
    for (int b = 0; b < 4; ++b) {
      h ^= (u >> (8 * b)) & 0xFF;
      h *= 1099511628211ull;
    }
  }
  return h;
}

namespace {
void put_le(std::vector<std::uint8_t>& o, std::uint64_t v, int bytes) {
  for (int b = 0; b < bytes; ++b) o.push_back(static_cast<std::uint8_t>(v >> (8 * b)));
}
std::uint64_t get_le(const std::uint8_t* p, int bytes) {
  std::uint64_t v = 0;
  for (int b = 0; b < bytes; ++b) v |= static_cast<std::uint64_t>(p[b]) << (8 * b);
  return v;
}
}  // namespace

Status kl_dump_write(const std::string& path, const KlDump& d) {
  std::vector<std::uint8_t> o;
  o.reserve(40 + d.tokens.size() * 24);
  for (char ch : std::string("PELKLD01")) o.push_back(static_cast<std::uint8_t>(ch));
  put_le(o, d.n_ctx, 4);
  put_le(o, d.chunks, 4);
  put_le(o, d.scored, 4);
  put_le(o, d.token_hash, 8);
  put_le(o, d.tokens.size(), 8);
  for (const KlToken& t : d.tokens) {
    std::uint64_t k;
    std::memcpy(&k, &t.kld, 8);
    put_le(o, k, 8);
    for (float f : {t.nll, t.nll_base, t.p_diff}) {
      std::uint32_t u;
      std::memcpy(&u, &f, 4);
      put_le(o, u, 4);
    }
    o.push_back(t.same_top ? 1 : 0);
    o.push_back(0);
    o.push_back(0);
    o.push_back(0);
  }
  std::ifstream exists(path, std::ios::binary);
  if (exists) return Status::fail(Code::invalid_argument, "the per-token file already exists (nothing is overwritten)");
  std::ofstream f(path, std::ios::binary);
  if (!f) return Status::fail(Code::io_error, "the per-token file could not be created");
  f.write(reinterpret_cast<const char*>(o.data()), static_cast<std::streamsize>(o.size()));
  if (!f) return Status::fail(Code::io_error, "the per-token file could not be written");
  return Status::good();
}

Result<KlDump> kl_dump_read(const std::string& path) {
  using R = Result<KlDump>;
  std::ifstream f(path, std::ios::binary);
  if (!f) return R::fail(Status::fail(Code::not_found, "the per-token file could not be opened"));
  std::vector<std::uint8_t> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (b.size() < 36 || std::memcmp(b.data(), "PELKLD01", 8) != 0) return R::fail(Status::fail(Code::corrupt, "not a Pelias per-token KL file"));
  KlDump d;
  d.n_ctx = static_cast<std::uint32_t>(get_le(b.data() + 8, 4));
  d.chunks = static_cast<std::uint32_t>(get_le(b.data() + 12, 4));
  d.scored = static_cast<std::uint32_t>(get_le(b.data() + 16, 4));
  d.token_hash = get_le(b.data() + 20, 8);
  const std::uint64_t n = get_le(b.data() + 28, 8);
  if (b.size() != 36 + n * 24) return R::fail(Status::fail(Code::corrupt, "the per-token file's size does not match its record count", 28));
  d.tokens.resize(static_cast<std::size_t>(n));
  for (std::uint64_t i = 0; i < n; ++i) {
    const std::uint8_t* p = b.data() + 36 + i * 24;
    KlToken& t = d.tokens[static_cast<std::size_t>(i)];
    const std::uint64_t k = get_le(p, 8);
    std::memcpy(&t.kld, &k, 8);
    float* fs[3] = {&t.nll, &t.nll_base, &t.p_diff};
    for (int j = 0; j < 3; ++j) {
      const auto u = static_cast<std::uint32_t>(get_le(p + 8 + 4 * j, 4));
      std::memcpy(fs[j], &u, 4);
    }
    t.same_top = p[20] != 0;
  }
  return R::good(std::move(d));
}

namespace {
// two-sided 95 % critical value of Student's t with df degrees of freedom (the chunk means are few: 100 chunks give
// 99, a quick 4-chunk check gives 3, where 1.96 would make the interval far too narrow)
double t975(std::size_t df) {
  static const double kTable[30] = {12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
                                    2.201,  2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
                                    2.080,  2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042};
  if (df == 0) return 0.0;
  if (df <= 30) return kTable[df - 1];
  const double z = 1.959964, v = static_cast<double>(df);  // Cornish-Fisher expansion, within 0.001 for df > 30
  return z + (z * z * z + z) / (4 * v) + (5 * std::pow(z, 5) + 16 * z * z * z + 3 * z) / (96 * v * v);
}
}  // namespace

Result<std::string> kl_compare(const KlDump& a, const KlDump& b, const std::string& name_a, const std::string& name_b) {
  using R = Result<std::string>;
  if (a.token_hash != b.token_hash || a.tokens.size() != b.tokens.size() || a.scored != b.scored || a.scored == 0)
    return R::fail(Status::fail(Code::invalid_argument, "the two files were not scored on the same tokens (different base, chunk count or context)"));
  const std::size_t n = a.tokens.size();
  const std::size_t per = a.scored;
  const std::size_t chunks = n / per;
  double sa = 0, sb = 0, sd = 0, sd2 = 0, snll = 0;
  std::size_t a_only = 0, b_only = 0, a_lower = 0;
  std::vector<double> cm(chunks, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    const KlToken& x = a.tokens[i];
    const KlToken& y = b.tokens[i];
    const double d = x.kld - y.kld;
    sa += x.kld;
    sb += y.kld;
    sd += d;
    sd2 += d * d;
    snll += static_cast<double>(x.nll) - static_cast<double>(y.nll);
    if (x.same_top && !y.same_top) ++a_only;
    if (!x.same_top && y.same_top) ++b_only;
    if (x.kld < y.kld) ++a_lower;
    if (i / per < chunks) cm[i / per] += d / static_cast<double>(per);
  }
  const double nn = static_cast<double>(n);
  const double mean = sd / nn;
  const double var_tok = (sd2 / nn - mean * mean) * nn / (nn - 1);
  const double se_tok = std::sqrt(std::max(0.0, var_tok) / nn);
  double cmean = 0, cvar = 0;
  for (double v : cm) cmean += v;
  cmean /= static_cast<double>(chunks);
  for (double v : cm) cvar += (v - cmean) * (v - cmean);
  cvar /= static_cast<double>(chunks > 1 ? chunks - 1 : 1);
  const double se_chunk = std::sqrt(cvar / static_cast<double>(chunks));
  if (chunks < 2) return R::fail(Status::fail(Code::invalid_argument, "a paired comparison needs at least 2 chunks (the standard error is taken between chunks)"));
  const double tc = t975(chunks - 1);
  const double lo = mean - tc * se_chunk, hi = mean + tc * se_chunk;
  char buf[2048];
  std::string o;
  std::snprintf(buf, sizeof buf, "paired on %zu scored tokens (%zu chunks of %zu)\n", n, chunks, per);
  o += buf;
  std::snprintf(buf, sizeof buf, "mean KLD   %s %.6f   %s %.6f\n", name_a.c_str(), sa / nn, name_b.c_str(), sb / nn);
  o += buf;
  std::snprintf(buf, sizeof buf, "difference (A - B) %+.6f   SE per token %.6f   SE per chunk (used) %.6f   95%% interval [%+.6f, %+.6f] (t %.3f, %zu df)   t-stat %.2f\n",
                mean, se_tok, se_chunk, lo, hi, tc, chunks - 1, se_chunk > 0 ? mean / se_chunk : 0.0);
  o += buf;
  std::snprintf(buf, sizeof buf, "tokens where A has the lower KL: %.2f %%; mean ln(PPL) difference %+.6f\n", 100.0 * static_cast<double>(a_lower) / nn, snll / nn);
  o += buf;
  const double disc = static_cast<double>(a_only + b_only);
  std::snprintf(buf, sizeof buf, "top token: right only in A %zu, only in B %zu (z %.2f)\n", a_only, b_only,
                disc > 0 ? (static_cast<double>(a_only) - static_cast<double>(b_only)) / std::sqrt(disc) : 0.0);
  o += buf;
  if (hi < 0) o += "VERDICT: A has less drift than B (the 95 % interval is below zero)\n";
  else if (lo > 0) o += "VERDICT: B has less drift than A (the 95 % interval is above zero)\n";
  else o += "VERDICT: no difference this test can resolve (the 95 % interval contains zero)\n";
  return R::good(o);
}

}  // namespace pelias::verify
