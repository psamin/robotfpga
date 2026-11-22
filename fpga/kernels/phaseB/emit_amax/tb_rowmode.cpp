// The unfused reference's QUANT_X: the row unit's real mode function (fpga/kernels/quantize/quant_mode.h
// qz::row_mode), built against the unpatched fpga/kernels/contract.h in its own translation unit (ctr::Cmd has the
// same layout in both headers; every contract.h function is static inline, so the two TUs do not clash).
#include <stdint.h>

#include "../../quantize/quant_mode.h"

uint32_t tb_row_mode(const ctr::Cmd& c, void* io_shifted) {
  return qz::row_mode(c, reinterpret_cast<const qz::word_t*>(io_shifted), reinterpret_cast<qz::word_t*>(io_shifted));
}
uint32_t tb_row_validate(const ctr::Cmd& c) { return ctr::validate(c); }
