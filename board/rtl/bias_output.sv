`timescale 1ns / 1ps
module bias_output #(parameter int unsigned LANES = 8) (
    input  logic [LANES-1:0][31:0] totals, biases,
    input  logic [4:0] shift,
    input  logic relu,
    output wire [LANES-1:0][7:0] y,
    output wire [LANES-1:0] overflow, range_error,
    output wire bad_shift
);
    assign bad_shift = (shift == 0 || shift == 31);
    for (genvar lane = 0; lane < LANES; lane++) begin : g_lane
        wire signed [32:0] sum;
        wire signed [7:0] quantized;
        wire unused_bad_shift;
        assign sum = $signed({totals[lane][31], totals[lane]}) +
                     $signed({biases[lane][31], biases[lane]});
        assign overflow[lane] = (sum < -33'sd2147483648 || sum > 33'sd2147483647);
        // Match the reference's reserved rounding headroom, not just INT32 fit.
        assign range_error[lane] = (sum < -33'sd2147483648 || sum >= 33'sd1073741824);
        requant u_requant (.acc(sum[31:0]), .shift(shift), .relu(relu),
                          .y(quantized), .bad_shift(unused_bad_shift));
        assign y[lane] = (overflow[lane] || range_error[lane] || bad_shift) ?
                        8'sd0 : quantized;
    end
endmodule
