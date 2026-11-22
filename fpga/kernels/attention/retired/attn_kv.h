// QUANT_W and QUANT_WT of the float_top integration contract (fpga/kernels/CONTRACT.md 1.1, 3.3, 5):
// FP32 key / value rows -> the int16 weight tilings the GEMM reads at run time.
//
//   QUANT_W  (OP 0x21)  KW: blob row n = key n, reduction index d < SEG; one scale per (key, segment)
//   QUANT_WT (OP 0x22)  VW: blob row c = channel c < SEG, reduction index = key n < NP; one scale per channel
//
// One mode function, quant_kv(cmd, xmem, ymem), for both opcodes: one call site in the sequencer's dispatch switch
// (case OP_QUANT_W: case OP_QUANT_WT:), INLINE off, so one hardware instance. It takes the top's io pointers
// (xmem = io read, ymem = io write; both registers are 0, so word index = byte address >> 4) and owns no m_axi
// adapter, no decode, no register. Precondition: ctr::validate_hw(cmd) == ME_OK (the sequencer checks it before every
// dispatch): every slot address is then < 2^40 and the host's validate() bounds every extent, so the address datapath
// is ap_uint<40> (CONTRACT.md 3.2). Counters are sized from the contract limits (N, NP, B <= 65535, K <= 16384).
//
// Arithmetic: the contract quantizer (quantize/quant_int16.h at its default QZ_RECIP_BITS = 1; this header does not
// change it): s = fl32(max(amax, 1e-30) / 32767), q = clamp(rne(fl32(x / s)), +-32767); the stored scale is
// qz::stored_scale: 1.0 for an all-zero row or channel, else s * 2^p4 (an exact exponent add: s >= 3.05e-35 and
// |p4| <= 8 keep it normal; a non-finite absmax is stored as is). Keys [N, NP) are zero rows with scale 1.0
// (QUANT_W) / zero columns (QUANT_WT). Non-finite inputs give undefined q but never a hang or a write outside
// the declared extents (tb_attn --unit checks both).
//
// Layouts (CONTRACT.md 1.1; W = SEG = p1 ? 64 : K, segment j = column / W, item b):
//   KW  word (n, d) of segment j at C + b C.stride + j p2 + ((n % 32) / 8) C.pitch + ((n / 32) W + d) 16, lane n % 8;
//       scale at F + b F.stride + j p3 + 4 n, for n < NP (NP % 32 == 0)
//   VW  word (c, n) of segment j at C + b C.stride + j p2 + ((c % 32) / 8) C.pitch + ((c / 32) NP + n) 16, lane c % 8;
//       scale at F + b F.stride + j p3 + 4 c, for c < W (W % 32 == 0), over the item's N keys; keys [N, NP) zero
//
// Hardware (one DATAFLOW pass, every loop II = 1):
//   reader  io read: one burst per row piece (KW: up to 4 heads = 64 words; VW: up to 2 heads = 32 words), each word
//           tagged (segment start, table index, last); KW: the absmax of each segment
//   scaler  s = scale_bits(amax) and R = qz::recip_unrolled(S), fully pipelined: one scale per clock
//   quant   4 lanes of qz::quant_elem (3 DSP each): 4 values per clock, 8-value int16 words out
//   writer  a 256-word LUTRAM tile (64 banks of 16 bits): KW transposes 8 keys x 256 values into 8-key words, VW keeps
//           8-channel words; one emit burst loop writes every run: KW 64-word runs (1 KB) per head and the 8 keys'
//           scales, VW 16-word runs per (port, group), the scale vectors of the table modes
//   QUANT_WT (and QUANT_W with p1 = 0 and K > 256) first scan the rows for the column (row) absmax: two reads.
// No multiplier or divider by a variable on any address path: bases advance by additions and shifts, and the loops
// that advance them are not flattened.
//
// Plain C++ on ap_int.h / hls_stream.h: the same code builds in Vitis HLS (attn_kv_top.cpp) and in g++
// (-I<Vitis_HLS>/include, tb_attn.cpp). ATTN_STREAM / ATTN_DEPTH / ATTN_DATAFLOW_THREADS let tb_attn run the
// DATAFLOW processes as threads on FIFOs bounded at the pragma depths (the deadlock model of CONTRACT.md 7).
#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

#include "../../contract.h"
#include "../../quantize/quant_int16.h"

#ifndef ATTN_STREAM
#define ATTN_STREAM hls::stream
#define ATTN_DEPTH(s, d)
#endif

namespace attn {

typedef ap_uint<128> Word;
typedef ap_uint<40> addr_t;  // byte address (< ctr::ADDR_LIMIT after validation)
using qz::abits_t;
using qz::RowScale;

constexpr uint32_t KW_KB = 8;    // keys per KW block: the 8 lanes of one port word
constexpr uint32_t KW_CW = 256;  // values per KW chunk: 4 heads, 64-word read bursts
constexpr uint32_t VW_KB = 16;   // keys per VW block: 16-word write runs
constexpr uint32_t VW_CW = 128;  // channels per VW chunk: 2 heads, 32-word read bursts
constexpr uint32_t TILE = 256;   // words of the writer's tile (KW 256 values x 8 keys, VW 16 groups x 16 keys)
constexpr uint32_t TBL = 32;     // table scales per lane (VW: 128 channels; KW long: 8 keys)
static_assert(KW_CW <= TILE && (VW_CW / 8) * VW_KB <= TILE && VW_CW <= 4 * TBL, "tile / table sizes");

// ---------------------------------------------------------------- one DATAFLOW pass
struct Pass {
  bool vw;     // QUANT_WT
  bool inl;    // scales inline (KW with W <= 256); else from the scan table
  bool seg64;  // W == 64 (p1 = 64), else W == K
  uint32_t K, W, N, NP;
  uint32_t key0;           // first key of the pass (KW long: one 8-key block per pass; else 0)
  uint32_t nblk;           // key blocks: KW 8 keys each; VW 16 keys each (the last one may hold 8)
  uint32_t nch, cwc, cwl;  // column chunks: count, width (values), width of the last
  uint32_t cc0;            // VW: first channel of the pass's chunk within its segment
  uint32_t nsc;            // table modes: scales of the pass (VW: the chunk's channels; KW long: 8 keys)
  ap_int<5> shift;         // p4 in [-8, 8]
  addr_t a_rows;           // A address of row key0 of the item (+ 4 c0 for VW)
  uint32_t apitch;
  addr_t c_base;  // KW: C + b stride + port / group offset of the pass's block 0; VW: C + b stride + j0 p2 + group
  uint32_t po0;   // KW: port of block 0 (0..3)
  uint32_t cpitch, segc;
  addr_t f_base;  // KW: F + b stride + 4 key0; VW: F + b stride + j0 p3 + 4 cc0
  uint32_t segf;
};

// one FP32 word with what the quantizer and the scaler need to know about it
struct Tw {
  Word x;
  ap_uint<5> idx;  // table index: VW word of the row piece (channels 4 idx ..), KW long key of the block
  bool seg0;       // inline: first word of a segment (its RowScale comes next)
  bool last;       // last word of the pass
};
struct Ta {  // inline: one segment's absmax
  abits_t a;
  bool last;
};

static inline addr_t times_rows(uint32_t rows, uint32_t pitch) {  // rows is 8 or 16
#pragma HLS INLINE
  return rows == VW_KB ? addr_t(addr_t(pitch) << 4) : addr_t(addr_t(pitch) << 3);
}

// Process 1: row pieces in bursts, tagged; KW inline: the absmax of every W-value segment.
static void reader(const Pass& P, const Word* xmem, ATTN_STREAM<Tw>& out, ATTN_STREAM<Ta>& amax) {
  addr_t blk_base = P.a_rows;
  uint32_t n0 = P.key0;
  const uint32_t segw = P.W / 4;  // words per segment (inline: segments never straddle a piece)
blocks:
  for (uint32_t blk = 0; blk < P.nblk; blk++) {
#pragma HLS LOOP_TRIPCOUNT max = 64
#pragma HLS LOOP_FLATTEN off
    const uint32_t left = P.NP - n0;
    const uint32_t rows = P.vw ? (left < VW_KB ? left : VW_KB) : KW_KB;
    const bool lastblk = blk == P.nblk - 1;
    addr_t chunk_base = blk_base;
  chunks:
    for (uint32_t ch = 0; ch < P.nch; ch++) {
#pragma HLS LOOP_TRIPCOUNT max = 3
#pragma HLS LOOP_FLATTEN off
      const bool lastch = ch == P.nch - 1;
      const uint32_t pw = (lastch ? P.cwl : P.cwc) / 4;
      addr_t ra = chunk_base;
    rows:
      for (uint32_t r = 0; r < rows; r++) {
#pragma HLS LOOP_TRIPCOUNT max = 16
#pragma HLS LOOP_FLATTEN off
        const bool data = n0 + r < P.N;
        const bool lastrow = lastblk && lastch && r == rows - 1;
        const uint64_t wi = (uint64_t)(ra >> 4);
        const ap_uint<5> kidx = r % 8;
        abits_t cur = 0;
        uint32_t ws = 0;
        if (data) {
        rd:
          for (uint32_t w = 0; w < pw; w++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 64
            Tw t;
            t.x = xmem[wi + w];
            t.idx = P.vw ? ap_uint<5>(w % TBL) : kidx;
            t.seg0 = ws == 0;
            t.last = lastrow && w == pw - 1;
            out.write(t);
            const abits_t m = qz::amax2(cur, qz::word_amax(t.x));
            const bool segend = ws == segw - 1;
            if (P.inl && segend) amax.write(Ta{m, t.last});
            cur = segend ? abits_t(0) : m;
            ws = segend ? 0u : ws + 1;
          }
        } else {
        zr:
          for (uint32_t w = 0; w < pw; w++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 64
            Tw t;
            t.x = 0;
            t.idx = P.vw ? ap_uint<5>(w % TBL) : kidx;
            t.seg0 = ws == 0;
            t.last = lastrow && w == pw - 1;
            out.write(t);
            const bool segend = ws == segw - 1;
            if (P.inl && segend) amax.write(Ta{abits_t(0), t.last});
            ws = segend ? 0u : ws + 1;
          }
        }
        ra += P.apitch;
      }
      chunk_base += 4 * P.cwc;
    }
    blk_base += times_rows(rows, P.apitch);
    n0 += rows;
  }
}

// Process 2: the scale unit, fully pipelined (qz::recip_unrolled: 25 compare-subtract stages). Inline: one scale per
// segment from the reader until `last`; table: P.nsc scales from the scan (am[i % 4][i / 4]). Out: the RowScale to
// quant, the stored scale (qz::stored_scale) to the writer.
static void scaler(const Pass& P, const abits_t am[4][TBL], ATTN_STREAM<Ta>& amax_in, ATTN_STREAM<RowScale>& rs_out,
                   ATTN_STREAM<uint32_t>& sc_out) {
  bool done = false;
  uint32_t i = 0;
sc:
  while (!done) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 4096
    abits_t a;
    if (P.inl) {
      const Ta t = amax_in.read();
      a = t.a;
      done = t.last;
    } else {
      a = am[i % 4][(i / 4) % TBL];
      done = i == P.nsc - 1;
    }
    RowScale v = qz::row_scale_from_s(qz::scale_bits(a));
    v.R = qz::recip_unrolled(v.S);
    rs_out.write(v);
    sc_out.write((uint32_t)qz::stored_scale(a, v.s_bits, P.shift));
    i++;
  }
}

// Process 3: 4 quantizations per clock; two FP32 words (8 values of one row) make one int16 word. Inline: one
// RowScale per segment for all lanes (read at the segment's first word); table: lane l of word idx = channel
// 4 idx + l (VW), or key idx (KW long).
static void quant(const Pass& P, ATTN_STREAM<Tw>& in, ATTN_STREAM<RowScale>& rs_in, ATTN_STREAM<Word>& out) {
  RowScale tbl[4][TBL];
#pragma HLS ARRAY_PARTITION variable = tbl complete dim = 1
#pragma HLS BIND_STORAGE variable = tbl type = ram_s2p impl = lutram
  if (!P.inl) {
  load:
    for (uint32_t i = 0; i < P.nsc; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 128
      const RowScale v = rs_in.read();
      for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
        if (P.vw ? (uint32_t)l == i % 4 : true) tbl[l][P.vw ? (i / 4) % TBL : i % TBL] = v;
      }
    }
  }
  RowScale cur = qz::row_scale_from_s(qz::ONE_BITS);
  ap_uint<64> lo = 0;
  bool odd = false, done = false;
qt:
  while (!done) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 8192
    const Tw t = in.read();
    if (P.inl && t.seg0) cur = rs_in.read();
    ap_uint<64> h = 0;
    for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
      const RowScale s = P.inl ? cur : tbl[l][t.idx];
      h.range(16 * l + 15, 16 * l) = ap_uint<16>(qz::quant_elem(qz::word_f32(t.x, l), s));
    }
    if (odd) out.write((Word(h) << 64) | Word(lo));
    lo = h;
    odd = !odd;
    done = t.last;
  }
}

// Process 4: the tile and every DDR write. Tile bank (lane l, column c), entry i holds lane l of tile word e = 8 i + c.
//   KW fill: int16 word ii of key r (values d = 8 ii .. + 7) -> banks (r, 0..7) entry ii (a transpose, no mux)
//   VW fill: int16 word m of key r (channels 8 m .. + 7) is tile word e = 16 m + r: banks (0..7, e % 8), entry e / 8
//   emit word e: banks (l, e % 8) entry e / 8, l = 0..7
// Scales: KW inline, captured at each segment's first int16 word (scale (key r, segment s) in scb[r % 4][2 s + r / 4],
// so a 4-key scale word is one entry per lane); table modes, the pass's scales first (sct[i % 4][i / 4]).
// One emit burst loop writes every run: source 0 = tile words, 1 = KW scale words, 2 = table scale words.
static void writer(const Pass& P, ATTN_STREAM<Word>& in, ATTN_STREAM<uint32_t>& sc, Word* ymem) {
  ap_uint<16> tile[8][8][TILE / 8];
#pragma HLS ARRAY_PARTITION variable = tile complete dim = 1
#pragma HLS ARRAY_PARTITION variable = tile complete dim = 2
#pragma HLS BIND_STORAGE variable = tile type = ram_s2p impl = lutram
  uint32_t scb[4][8], sct[4][TBL];
#pragma HLS ARRAY_PARTITION variable = scb complete dim = 1
#pragma HLS ARRAY_PARTITION variable = sct complete dim = 1
#pragma HLS BIND_STORAGE variable = scb type = ram_s2p impl = lutram
#pragma HLS BIND_STORAGE variable = sct type = ram_s2p impl = lutram
  if (!P.inl) {
  tload:
    for (uint32_t i = 0; i < P.nsc; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 128
      const uint32_t v = sc.read();
      for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
        if ((uint32_t)l == i % 4) sct[l][(i / 4) % TBL] = v;
      }
    }
  }
  const addr_t W16 = addr_t(P.W) << 4;  // bytes of one KW group in a port array (W words)
  const uint32_t segq = P.W / 8;        // int16 words per segment (KW inline)
  // table scale runs, written with the pass's first chunk: VW W = 64 with a 128-channel chunk: two segments
  const bool two = P.vw && P.seg64 && P.cwc > 64;
  const uint32_t tab_runs = P.inl ? 0u : two ? 2u : 1u;
  const uint32_t tab_len = P.vw ? (two ? 16u : P.cwc / 4) : KW_KB / 4;
  addr_t c_blk = P.c_base, f_blk = P.f_base;
  uint32_t po = P.po0, n0 = 0;
blocks:
  for (uint32_t blk = 0; blk < P.nblk; blk++) {
#pragma HLS LOOP_TRIPCOUNT max = 64
#pragma HLS LOOP_FLATTEN off
    const uint32_t left = P.NP - n0;
    const uint32_t rows = P.vw ? (left < VW_KB ? left : VW_KB) : KW_KB;
    addr_t c_ch = c_blk, f_ch = f_blk;  // KW: + j0 p2 / + j0 p3 of the chunk's first segment
    uint32_t c0 = 0;                    // KW long: first column of the chunk
  chunks:
    for (uint32_t ch = 0; ch < P.nch; ch++) {
#pragma HLS LOOP_TRIPCOUNT max = 3
#pragma HLS LOOP_FLATTEN off
      const uint32_t cw = ch == P.nch - 1 ? P.cwl : P.cwc;
      const uint32_t qw = cw / 8;  // int16 words per row piece
      // ---- fill (+ KW inline: one scale at each segment's first word)
      uint32_t ii = 0, r = 0, sq = 0, ks = 0;
      const uint32_t nfill = rows == VW_KB ? qw << 4 : qw << 3;
    fill:
      for (uint32_t i = 0; i < nfill; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 256
#pragma HLS DEPENDENCE variable = tile inter false
        const Word q = in.read();
        const uint32_t e = 16 * ii + r;  // VW tile word
        for (int l = 0; l < 8; l++) {
#pragma HLS UNROLL
          for (int c = 0; c < 8; c++) {
#pragma HLS UNROLL
            const bool we = P.vw ? (e % 8) == (uint32_t)c : r == (uint32_t)l;
            const ap_uint<5> ad = P.vw ? ap_uint<5>((e / 8) % (TILE / 8)) : ap_uint<5>(ii % (TILE / 8));
            const ap_uint<16> d = P.vw ? ap_uint<16>(q.range(16 * l + 15, 16 * l)) : ap_uint<16>(q.range(16 * c + 15, 16 * c));
            if (we) tile[l][c][ad] = d;
          }
        }
        if (P.inl && sq == 0) {
          const uint32_t v = sc.read();
          for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
            if ((uint32_t)l == r % 4) scb[l][(2 * ks + r / 4) % 8] = v;
          }
        }
        const bool seg_end = sq == segq - 1;
        sq = seg_end ? 0u : sq + 1;
        if (ii == qw - 1) {
          ii = 0;
          r++;
          ks = 0;
          sq = 0;
        } else {
          ii++;
          if (seg_end) ks++;
        }
      }
      // ---- emit: the runs of this (block, chunk)
      const bool first = blk == 0 && ch == 0;
      const uint32_t ntab = first ? tab_runs : 0u;
      const uint32_t segs = P.inl ? (P.seg64 ? cw / 64 : 1u) : 1u;
      const uint32_t nruns = ntab + (P.vw ? qw : (P.inl ? 2 * segs : segs));
      // run state
      addr_t fa_t = P.f_base;                                                     // table scale runs
      addr_t oa = P.inl ? c_ch : addr_t(c_ch + (addr_t(c0) << 4)), fa = f_ch;  // KW data / scale runs
      uint32_t e0 = 0, ks2 = 0;
      uint32_t cc = P.cc0;  // VW: channel within the segment
      addr_t sbase = c_blk + (addr_t(n0) << 4), pofs = 0, gofs = 0;
    runs:
      for (uint32_t u = 0; u < nruns; u++) {
#pragma HLS LOOP_TRIPCOUNT max = 18
#pragma HLS LOOP_FLATTEN off
        addr_t ra;
        uint32_t len, eb;
        ap_uint<2> src;
        if (u < ntab) {  // a table scale vector (or the 8 keys' scales of a KW long block)
          ra = fa_t;
          len = tab_len;
          src = 2;
          eb = u * 16;
          fa_t += P.segf;
        } else if (P.vw) {  // VW group m: keys n0 .. n0 + rows - 1
          ra = sbase + pofs + gofs;
          len = rows;
          src = 0;
          eb = e0;
          e0 += 16;
          cc += 8;
          if (cc == P.W) {  // next segment
            cc = 0;
            sbase += P.segc;
            pofs = 0;
            gofs = 0;
          } else if (cc % 32 == 0) {  // next group
            pofs = 0;
            gofs += addr_t(P.NP) << 4;
          } else {  // next port
            pofs += P.cpitch;
          }
        } else if (!P.inl || ((u - ntab) % 2) == 0) {  // KW data run of segment ks2
          ra = oa;
          len = P.inl ? P.W : cw;
          src = 0;
          eb = e0;
          e0 += len;
          oa += P.segc;
        } else {  // KW inline: the block's 8 key scales of segment ks2
          ra = fa;
          len = 2;
          src = 1;
          eb = 2 * ks2;
          ks2++;
          fa += P.segf;
        }
        const uint64_t wi = (uint64_t)(ra >> 4);
      emit:
        for (uint32_t w = 0; w < len; w++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 64
          const uint32_t e = eb + w;
          Word o = 0;
          if (src == 0) {
            for (int l = 0; l < 8; l++) {
#pragma HLS UNROLL
              ap_uint<16> v = 0;
              for (int c = 0; c < 8; c++) {
#pragma HLS UNROLL
                if (e % 8 == (uint32_t)c) v = tile[l][c][(e / 8) % (TILE / 8)];
              }
              o.range(16 * l + 15, 16 * l) = v;
            }
          } else {
            for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
              o.range(32 * l + 31, 32 * l) = src == 1 ? scb[l][e % 8] : sct[l][e % TBL];
            }
          }
          ymem[wi + w] = o;
        }
      }
      if (!P.vw && P.seg64) {  // KW: the next chunk starts 4 segments later
        c_ch += addr_t(P.segc) << 2;
        f_ch += addr_t(P.segf) << 2;
      }
      c0 += cw;
    }
    if (!P.vw) {  // next block: next port, every 4 blocks the next group
      if (po == 3) {
        po = 0;
        c_blk += W16 - 3 * addr_t(P.cpitch);
      } else {
        po++;
        c_blk += P.cpitch;
      }
      f_blk += 4 * KW_KB;
    }
    n0 += rows;
  }
}

static void kv_pass(const Pass& P, const abits_t am[4][TBL], const Word* xmem, Word* ymem) {
#pragma HLS DATAFLOW
#pragma HLS STABLE variable = am
  ATTN_STREAM<Tw> raw;
  ATTN_STREAM<Word> qw;
  ATTN_STREAM<Ta> amax;
  ATTN_STREAM<RowScale> rs;
  ATTN_STREAM<uint32_t> sc;
  // raw: a 64-word row piece + the scaler latency; qw: two KW chunks (the writer's emit overlaps the next fill);
  // sc: a VW chunk's 128 table scales (the writer reads them while quant loads its table)
#pragma HLS STREAM variable = raw depth = 128
#pragma HLS STREAM variable = qw depth = 512
#pragma HLS STREAM variable = amax depth = 8
#pragma HLS STREAM variable = rs depth = 8
#pragma HLS STREAM variable = sc depth = 128
#pragma HLS BIND_STORAGE variable = raw type = fifo impl = bram
#pragma HLS BIND_STORAGE variable = qw type = fifo impl = bram
#pragma HLS BIND_STORAGE variable = amax type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = rs type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = sc type = fifo impl = lutram
  ATTN_DEPTH(raw, 128);
  ATTN_DEPTH(qw, 512);
  ATTN_DEPTH(amax, 8);
  ATTN_DEPTH(rs, 8);
  ATTN_DEPTH(sc, 128);
#ifdef ATTN_DATAFLOW_THREADS
  ATTN_DATAFLOW_THREADS(reader(P, xmem, raw, amax), scaler(P, am, amax, rs, sc), quant(P, raw, rs, qw),
                        writer(P, qw, sc, ymem));
#else
  reader(P, xmem, raw, amax);
  scaler(P, am, amax, rs, sc);
  quant(P, raw, rs, qw);
  writer(P, qw, sc, ymem);
#endif
}

// Pre-pass of QUANT_WT (column absmax over the item's N keys, one chunk of up to 128 channels) and of QUANT_W with
// W = K > 256 (row absmax of the 8 keys of one block): am[i % 4][i / 4] = the absmax of scale i of the pass.
static void scan(const Pass& P, const Word* xmem, abits_t am[4][TBL]) {
clr:
  for (uint32_t w = 0; w < TBL; w++) {
#pragma HLS PIPELINE II = 1
    for (int l = 0; l < 4; l++) am[l][w] = 0;
  }
  const uint32_t nrows = P.vw ? P.N : KW_KB;
  const uint32_t nch = P.vw ? 1u : P.nch;
  addr_t ra = P.a_rows;
rows:
  for (uint32_t r = 0; r < nrows; r++) {
#pragma HLS LOOP_TRIPCOUNT max = 256
#pragma HLS LOOP_FLATTEN off
    abits_t cur = 0;
    const bool data = P.vw || P.key0 + r < P.N;
    addr_t ca = ra;
  chunks:
    for (uint32_t ch = 0; ch < nch; ch++) {
#pragma HLS LOOP_TRIPCOUNT max = 64
#pragma HLS LOOP_FLATTEN off
      const uint32_t pw = (ch == nch - 1 ? P.cwl : P.cwc) / 4;  // VW chunks are >= 32 channels: pw >= 8
      const uint64_t wi = (uint64_t)(ca >> 4);
      if (data) {
      col:
        for (uint32_t w = 0; w < pw; w++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 64
#pragma HLS DEPENDENCE variable = am inter distance = 8 true
          const Word x = xmem[wi + w];
          if (P.vw) {
            for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
              am[l][w % TBL] = qz::amax2(am[l][w % TBL], qz::abs_bits(qz::word_f32(x, l)));
            }
          } else {
            cur = qz::amax2(cur, qz::word_amax(x));
          }
        }
      }
      ca += 4 * P.cwc;
    }
    if (!P.vw) am[r % 4][r / 4] = cur;
    ra += P.apitch;
  }
}

// ---------------------------------------------------------------- the mode function
// QUANT_W / QUANT_WT. Precondition: ctr::validate_hw(c) == ME_OK. Returns ME_OK after the last write response.
static uint32_t quant_kv(const ctr::Cmd& c, const Word* xmem, Word* ymem) {
#pragma HLS INLINE off
  const ctr::Slot &A = c.s[ctr::SLOT_A], &C = c.s[ctr::SLOT_C], &F = c.s[ctr::SLOT_F];
  Pass P;
  P.vw = c.op == ctr::OP_QUANT_WT;
  P.seg64 = c.p[ctr::P_SEG] != 0;
  P.K = c.K;
  P.W = P.seg64 ? (uint32_t)ctr::SEG_HEAD : c.K;
  P.N = c.N;
  P.NP = c.p[ctr::P_NP] ? c.p[ctr::P_NP] : c.N;
  P.shift = ap_int<5>((int32_t)c.p[ctr::P_SHIFT]);
  P.apitch = A.pitch;
  P.cpitch = C.pitch;
  P.segc = c.p[ctr::P_SEGC];
  P.segf = c.p[ctr::P_SEGF];
  const bool kw_long = !P.vw && P.W > KW_CW;
  P.inl = !P.vw && !kw_long;
  // KW chunks: 4 heads (W = 64), the whole row (W = K <= 256) or 256 values (W = K > 256)
  const bool kw256 = P.seg64 || kw_long;
  const uint32_t kw_nch = kw256 ? (P.K + KW_CW - 1) / KW_CW : 1u;
  const uint32_t kw_cwl = kw256 ? P.K - ((kw_nch - 1) * KW_CW) : P.K;
  const uint32_t nkb = P.NP / KW_KB;
  const uint32_t nvch = (P.K + VW_CW - 1) / VW_CW;
  const uint32_t npass = P.vw ? nvch : kw_long ? nkb : 1u;
  static abits_t am[4][TBL];
#pragma HLS ARRAY_PARTITION variable = am complete dim = 1
#pragma HLS BIND_STORAGE variable = am type = ram_2p impl = lutram
  addr_t a_item = A.addr, c_item = C.addr, f_item = F.addr;
items:
  for (uint32_t b = 0; b < c.B; b++) {
#pragma HLS LOOP_TRIPCOUNT max = 24
#pragma HLS LOOP_FLATTEN off
    addr_t a_rows = a_item;               // KW long: row 8 o
    addr_t c_kw = c_item, f_kw = f_item;  // KW long: block o's port / group base, scale base
    addr_t c_vw = c_item, f_vw = f_item;  // VW: + j0 p2 (+ group offset), + j0 p3 of the pass's chunk
    uint32_t po = 0, c0 = 0;
  passes:
    for (uint32_t o = 0; o < npass; o++) {
#pragma HLS LOOP_TRIPCOUNT max = 8
#pragma HLS LOOP_FLATTEN off
      P.key0 = kw_long ? KW_KB * o : 0u;
      if (P.vw) {
        const uint32_t cw = P.K - c0 < VW_CW ? P.K - c0 : VW_CW;
        P.nch = 1;
        P.cwc = cw;
        P.cwl = cw;
        P.cc0 = P.seg64 ? 0u : c0;
        P.nblk = (P.NP + VW_KB - 1) / VW_KB;
        P.nsc = cw;
        P.a_rows = a_item + (addr_t(c0) << 2);
        P.c_base = c_vw;
        P.f_base = f_vw + (addr_t(P.cc0) << 2);
        P.po0 = 0;
      } else {
        P.nch = kw_nch;
        P.cwc = kw256 ? KW_CW : P.K;
        P.cwl = kw_cwl;
        P.cc0 = 0;
        P.nblk = kw_long ? 1u : nkb;
        P.nsc = KW_KB;
        P.a_rows = a_rows;
        P.c_base = c_kw;
        P.f_base = f_kw;
        P.po0 = po;
      }
      if (!P.inl) scan(P, xmem, am);
      kv_pass(P, am, xmem, ymem);
      // next pass: VW the next 128 channels; KW long the next 8-key block
      c0 += VW_CW;
      if (P.seg64) {  // W = 64: the next VW chunk starts two segments later
        c_vw += addr_t(P.segc) << 1;
        f_vw += addr_t(P.segf) << 1;
      } else {  // W = K: four 32-channel groups later
        c_vw += addr_t(P.NP) << 6;
      }
      a_rows += addr_t(A.pitch) << 3;
      f_kw += 4 * KW_KB;
      if (po == 3) {
        po = 0;
        c_kw += (addr_t(P.W) << 4) - 3 * addr_t(C.pitch);
      } else {
        po++;
        c_kw += C.pitch;
      }
    }
    a_item += A.stride;
    c_item += C.stride;
    f_item += F.stride;
  }
  return ctr::ME_OK;
}

}  // namespace attn
