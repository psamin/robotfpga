`timescale 1ns / 1ps
module requant (
    input  logic signed [31:0] acc,
    input  logic [4:0] shift,
    input  logic relu,
    output logic signed [7:0] y,
    output logic bad_shift
);
    logic signed [32:0] rounded, scaled;
    always_comb begin
        bad_shift = (shift == 0 || shift == 31);
        rounded = 33'sd0;
        scaled = 33'sd0;
        y = 8'sd0;
        if (!bad_shift) begin
            // An extra bit prevents rounding from wrapping at INT32_MAX.
            rounded = $signed({acc[31], acc}) + (33'sd1 << (shift - 1'b1));
            scaled = rounded >>> shift;
            if (scaled > 33'sd127) y = 8'sd127;
            else if (relu && scaled < 0) y = 8'sd0;
            else if (scaled < -33'sd127) y = -8'sd127;
            else y = scaled[7:0];
        end
    end
endmodule
