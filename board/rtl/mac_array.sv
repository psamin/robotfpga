`timescale 1ns / 1ps
module mac_array #(
    parameter int unsigned LANES = 8
) (
    input  logic clk, reset, clear,
    input  logic [LANES-1:0] enable,
    input  logic [LANES-1:0][7:0] a, b,
    output wire [LANES-1:0][31:0] acc
);
    for (genvar lane = 0; lane < LANES; lane++) begin : g_lane
        mac u_mac (
            .clk(clk), .reset(reset), .clear(clear), .enable(enable[lane]),
            .a(a[lane]), .b(b[lane]), .acc(acc[lane])
        );
    end
endmodule
