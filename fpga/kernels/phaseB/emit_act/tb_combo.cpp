// emit_act + emit_amax merged (build/merge from merge_emit_amax.py): the act GEMM writes the activated FP32 rows
// and their absmax (P_ACT + P_AMX_SG), the next GEMM quantizes them on load (P_XQ). Against the contract 1.0
// chain, on one DDR image per chain (same addresses, same junk):
//   unfused: GEMM1 (natural blob) -> Y; GELU / SILU_MUL / SILU (F_QOUT) -> Q, S; GEMM2 (int16 Q, s_tok S) -> Z
//   fused:   GEMM1 (P_ACT, P_AMX_SG = sg; interleaved blob for SILU_MUL) -> O + H; GEMM2 (P_XQ, A = O, F = H) -> Z
// Checks: every command valid (merged contract.h: validate and validate_hw); Z equal bit for bit; O == the direct
// reference (ctr::ref::gemm_out + gs::h_bits + fmul_hw); H == the absmax bits of O per (row, segment) computed
// here; the fused image equal to the initial image outside O, H and Z (nothing else written).
// Also re-runs tb_emit_act's fused == unfused cases on the merged tree (P_AMX_SG = P_XQ = 0).
//   g++ -std=c++14 -O2 -ffp-contract=off -DFSTREAM_ARITH=16 -DGS_NSB=10 -I<merge>/hls -I<merge>/kernels/quantize
//       -I<Vitis>/include tb_combo.cpp
//   tb_combo [random cases] [seed]
#define EA_TB_NO_MAIN 1
#include "tb_emit_act.cpp"

#ifndef P_AMX_SG_PRESENT
static_assert(ctr::P_AMX_SG == 7 && ctr::P_XQ == 8 && ctr::P_ACT == 6, "merged contract: p6 P_ACT, p7 P_AMX_SG, p8 P_XQ");
#endif

struct Combo {
  uint32_t M2;  // GEMM2 output columns (multiple of 32); its K = F
  uint32_t sg;  // groups per absmax segment (divides G1 = M / 32)
  std::vector<int16_t> W2;  // [b][M2][F]
  std::vector<float> s2;    // [b][M2]
};

static bool run_combo(Case& cs, Combo& cb, uint64_t base, bool verbose) {
  Img m0;
  m0.base = base;
  const Lay L = layout(m0, cs);
  const uint32_t B = cs.B, N = cs.N, F = cs.F, M2 = cb.M2, G1 = cs.M / 32, parts = G1 / cb.sg;
  // GEMM2 blob, s_row, Z, and the absmax parts H
  const uint64_t port = (uint64_t)(M2 / 32) * F * 16, wp = port + 16 * rnd(2), ws = 4 * wp + 16 * rnd(2);
  const ctr::Slot W2 = {m0.alloc(B * ws), (uint32_t)wp, B > 1 ? (uint32_t)ws : 0};
  const uint64_t e2s = 4 * M2 + 16 * rnd(2);
  const ctr::Slot E2 = {m0.alloc(B * e2s), 0, B > 1 ? (uint32_t)e2s : 0};
  const uint64_t zp = 4 * M2 + 16 * rnd(3), zs = N * zp + 16 * rnd(3);
  const ctr::Slot Z = {m0.alloc(B * zs), (uint32_t)zp, B > 1 ? (uint32_t)zs : 0};
  const uint64_t hp = 4 * ctr::pad4(N) + 16 * rnd(3), hs = parts * hp + 16 * rnd(3);
  const ctr::Slot H = {m0.alloc(B * hs), parts > 1 ? (uint32_t)hp : 0, B > 1 ? (uint32_t)hs : 0};
  put_operands(m0, cs, L);
  cb.W2.resize((size_t)B * M2 * F);
  cb.s2.resize((size_t)B * M2);
  for (auto& v : cb.W2) v = q_of(0);
  for (auto& v : cb.s2) v = (float)ldexp(1.0 + urand(), -20 - (int)rnd(6));
  for (uint32_t b = 0; b < B; b++) {
    for (uint32_t r = 0; r < M2; r++) {
      for (uint32_t k = 0; k < F; k++)
        m0.set_i16(ctr::w_word_addr(W2, b, r, k, F) + 2 * ctr::w_lane(r), cb.W2[((size_t)b * M2 + r) * F + k]);
      m0.set_f32(E2.addr + b * E2.stride + 4 * r, cb.s2[(size_t)b * M2 + r]);
    }
  }
  // commands
  const ctr::Cmd g1u = gemm_cmd(cs, L, false), au = act_cmd(cs, L, true);
  ctr::Cmd g1f = gemm_cmd(cs, L, true);
  g1f.p[ctr::P_AMX_SG] = cb.sg;
  g1f.s[ctr::SLOT_H] = H;
  ctr::Cmd g2u;
  memset(&g2u, 0, sizeof g2u);
  g2u.op = ctr::OP_GEMM;
  g2u.M = M2;
  g2u.K = F;
  g2u.N = N;
  g2u.B = B;
  g2u.s[ctr::SLOT_A] = L.Q;
  g2u.s[ctr::SLOT_B] = W2;
  g2u.s[ctr::SLOT_C] = Z;
  g2u.s[ctr::SLOT_E] = E2;
  g2u.s[ctr::SLOT_F] = L.S;
  ctr::Cmd g2f = g2u;
  g2f.s[ctr::SLOT_A] = L.O;
  g2f.s[ctr::SLOT_F] = H;
  g2f.p[ctr::P_XQ] = parts;
  const ctr::Cmd* all[] = {&g1u, &au, &g2u, &g1f, &g2f};
  const char* nm[] = {"GEMM1", "ACT+QOUT", "GEMM2", "GEMM1+ACT+AMX", "GEMM2+XQ"};
  for (int i = 0; i < 5; i++) {
    const uint32_t v = ctr::validate(*all[i]), vh = ctr::validate_hw(*all[i]);
    if (v != ctr::ME_OK || vh != ctr::ME_OK) {
      printf("FAIL %s: %s invalid (validate %u, validate_hw %u)\n", cs.name.c_str(), nm[i], v, vh);
      return false;
    }
  }
  Img mu = m0, mf = m0;
  uint32_t r = 0;
  {
    IoWord* io = mu.io();
    r |= fgemm::gemm_unit(g1u, mu.hp(), mu.hp(), mu.hp(), mu.hp(), (const IoWord*)io, io);
    r |= gs::act_mode(au, (const IoWord*)io, io);
    r |= fgemm::gemm_unit(g2u, mu.hp(), mu.hp(), mu.hp(), mu.hp(), (const IoWord*)io, io);
  }
  {
    IoWord* io = mf.io();
    r |= fgemm::gemm_unit(g1f, mf.hp(), mf.hp(), mf.hp(), mf.hp(), (const IoWord*)io, io);
    r |= fgemm::gemm_unit(g2f, mf.hp(), mf.hp(), mf.hp(), mf.hp(), (const IoWord*)io, io);
  }
  if (r != ctr::ME_OK) {
    printf("FAIL %s: a unit returned %u\n", cs.name.c_str(), r);
    return false;
  }
  // Z equal; O == direct reference; H == absmax of O per (row, segment)
  size_t zd = 0, od = 0, hd = 0, other = 0;
  for (uint32_t b = 0; b < B; b++)
    for (uint32_t n = 0; n < N; n++) {
      for (uint32_t c = 0; c < M2; c++) {
        const uint64_t a = ctr::row_addr(Z, b, n) + 4 * c;
        zd += mu.u32(a) != mf.u32(a);
      }
      const uint32_t cps = cs.act == ea::ACT_SILU_MUL ? 16 * cb.sg : 32 * cb.sg;  // output columns per segment
      for (uint32_t p = 0; p < parts; p++) {
        uint32_t am = 0;
        for (uint32_t c = p * cps; c < (p + 1) * cps; c++) {
          const uint32_t mag = mf.u32(ctr::row_addr(L.O, b, n) + 4 * c) & 0x7FFFFFFFu;
          am = mag > am ? mag : am;
        }
        hd += mf.u32(H.addr + b * H.stride + (uint64_t)p * H.pitch + 4 * n) != am;
      }
      for (uint32_t c = 0; c < F; c++)
        od += mf.u32(ctr::row_addr(L.O, b, n) + 4 * c) != f_bits(ref_out(cs, b, n, c));
    }
  // the fused image: unchanged outside O, H, Z
  auto in = [&](size_t j, const ctr::Slot& s, uint64_t item_bytes) {
    const uint64_t a = s.addr - base;
    return j >= a && j < a + (uint64_t)B * (s.stride ? s.stride : item_bytes);
  };
  for (size_t j = 0; j < m0.b.size(); j++) {
    if (in(j, L.O, (uint64_t)N * L.O.pitch) || in(j, H, (uint64_t)parts * (H.pitch ? H.pitch : 4 * ctr::pad4(N))) ||
        in(j, Z, (uint64_t)N * Z.pitch))
      continue;
    other += mf.b[j] != m0.b[j];
  }
  const bool ok = zd == 0 && od == 0 && hd == 0 && other == 0;
  if (!ok || verbose)
    printf("%s %s: act %d M=%u K=%u N=%u B=%u F=%u -> M2=%u, sg=%u (%u parts): Z %zu differ, O %zu vs ref, H %zu, "
           "%zu bytes written elsewhere\n",
           ok ? "ok  " : "FAIL", cs.name.c_str(), cs.act, cs.M, cs.K, N, B, F, M2, cb.sg, parts, zd, od, hd, other);
  return ok;
}

int main(int argc, char** argv) {
  const int cases = argc > 1 ? atoi(argv[1]) : 120;
  rng.seed(argc > 2 ? strtoull(argv[2], nullptr, 0) : 11);
  const uint64_t base = 0x800000000ull;
  int pass = 0, fail = 0;
  // the four -12392 chains: fc1 + gelu -> fc2, LM / expert gate|up + silu*up -> down, time_in + silu -> time_out
  struct R {
    const char* n;
    int act;
    uint32_t M, K, N, M2;
    bool bias;
    int d;
  } rs[] = {{"siglip_fc1_gelu_fc2", ea::ACT_GELU, 3072, 768, 512, 768, true, 0},
            {"lm_gu_silu_down", ea::ACT_SILU_MUL, 5120, 960, 45, 960, false, 0},
            {"expert_gu_silu_down", ea::ACT_SILU_MUL, 4096, 720, 50, 736, false, 0},
            {"time_in_silu_out", ea::ACT_SILU, 736, 720, 50, 736, false, 2}};
  for (auto& x : rs) {
    Case cs = real_case(x.n, x.act, x.M, x.K, x.N, x.bias, x.d);
    Combo cb;
    cb.M2 = x.M2;
    cb.sg = x.M / 32;  // one segment: the whole row
    if (run_combo(cs, cb, base, true)) pass++;
    else fail++;
  }
  for (int i = 0; i < cases && fail < 6; i++) {
    Case cs;
    random_case(cs, i);
    Combo cb;
    cb.M2 = 32 * (uint32_t)(1 + rnd(6));
    const uint32_t G1 = cs.M / 32;
    const uint32_t opts[] = {G1, G1 % 2 == 0 ? G1 / 2 : G1, 1};
    cb.sg = opts[rnd(3)];
    if (G1 / cb.sg > ctr::AMX_NS_MAX) cb.sg = G1;
    if (run_combo(cs, cb, base, false)) pass++;
    else fail++;
  }
  // tb_emit_act's own cases on the merged tree (the emit_amax parameters 0)
  Stat st;
  int p2 = 0, f2 = 0;
  for (int i = 0; i < cases && f2 < 6; i++) {
    Case cs;
    random_case(cs, 1000 + i);
    if (run_case(cs, base, st, false)) p2++;
    else f2++;
  }
  printf("%s: merged emit_act + emit_amax: %d of %d chains (GEMM+ACT+AMX -> GEMM+XQ) == contract 1.0 chain (GEMM -> "
         "ACT F_QOUT -> GEMM) bit for bit; %d of %d emit_act-only cases fused == unfused\n",
         fail + f2 ? "FAIL" : "PASS", pass, pass + fail, p2, p2 + f2);
  return fail + f2 ? 1 : 0;
}
