// provenance: original (test vectors from FIPS 180-4 / NIST examples, public)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include <string>
#include <vector>

#include "io/sha256.h"
#include "pelias_test.h"

using pelias::io::Sha256;

TEST(sha256_known_vectors) {
  CHECK_EQ(Sha256::hex_of("", 0), std::string("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
  CHECK_EQ(Sha256::hex_of("abc", 3), std::string("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
  const std::string two = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  CHECK_EQ(Sha256::hex_of(two.data(), two.size()), std::string("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"));
}

TEST(sha256_streaming_matches_one_shot) {
  std::vector<std::uint8_t> data(1000003);
  for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<std::uint8_t>(i * 7 + (i >> 3));
  const std::string one = Sha256::hex_of(data.data(), data.size());
  Sha256 h;
  std::size_t pos = 0;
  const std::size_t chunks[] = {1, 63, 64, 65, 1000, 4096, 100000};
  int k = 0;
  while (pos < data.size()) {
    const std::size_t n = std::min(chunks[k++ % 7], data.size() - pos);
    h.update(data.data() + pos, n);
    pos += n;
  }
  CHECK_EQ(Sha256::hex(h.finish()), one);
  // one million 'a' (NIST vector; checked against python hashlib 2026-10-26)
  std::string a(1000000, 'a');
  CHECK_EQ(Sha256::hex_of(a.data(), a.size()), std::string("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"));
}
