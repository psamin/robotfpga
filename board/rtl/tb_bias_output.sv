`timescale 1ns / 1ps
module bias_output_case #(parameter int LANES = 8)(output bit done);
    logic [LANES-1:0][31:0] totals, biases;
    logic [4:0] shift;
    logic relu;
    wire [LANES-1:0][7:0] y;
    wire [LANES-1:0] overflow, range_error;
    wire bad_shift;
    int checks = 0;
    logic [31:0] rng = 32'h74c0ffee ^ LANES;
    bias_output #(.LANES(LANES)) dut (.*);
    function automatic logic [31:0] random_word();
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        return rng;
    endfunction

    task automatic check(input int mode, s, input bit r);
        longint signed av, bv, sum, divisor, numerator, result;
        longint signed expected [LANES];
        bit ov [LANES], invalid [LANES];
        logic [31:0] ra, rb;
        shift = s; relu = r;
        for (int lane = 0; lane < LANES; lane++) begin
            case (mode)
                0: begin av = 100 + 7 * lane; bv = -3 * lane; end
                1: begin av = 64'sd2147483647; bv = 1; end
                2: begin av = -64'sd2147483648; bv = -1; end
                3: begin av = -64'sd2147483648; bv = av; end
                4: begin av = 64'sd2147483647; bv = av; end
                5: begin av = 64'sd1073741823; bv = 0; end
                6: begin av = 64'sd1073741824; bv = 0; end
                7: begin av = -64'sd2147483648; bv = 0; end
                8: begin av = 64'sd2147483647; bv = -64'sd1073741824; end
                9: begin av = -64'sd2147483648; bv = 64'sd2147483647; end
                default: begin
                    ra = random_word(); rb = random_word();
                    if (mode == 10) begin av = $signed(ra); bv = $signed(rb); end
                    else begin av = int'(ra[15:0]) - 32768; bv = int'(rb[15:0]) - 32768; end
                end
            endcase
            totals[lane] = av; biases[lane] = bv;
            sum = av + bv;
            ov[lane] = (sum < -64'sd2147483648 || sum > 64'sd2147483647);
            invalid[lane] = (sum < -64'sd2147483648 || sum >= 64'sd1073741824);
            result = 0;
            if (!invalid[lane] && s >= 1 && s <= 30) begin
                divisor = 64'sd1 << s; numerator = sum + divisor / 2;
                result = numerator < 0 ? -((-numerator + divisor - 1) / divisor) : numerator / divisor;
                if (result < (r ? 0 : -127)) result = r ? 0 : -127;
                if (result > 127) result = 127;
            end
            expected[lane] = result;
        end
        #1;
        if (bad_shift !== (s == 0 || s == 31)) $fatal(1, "bad_shift mismatch");
        for (int lane = 0; lane < LANES; lane++) begin
            checks++;
            if ($signed(y[lane]) !== expected[lane] || overflow[lane] !== ov[lane] ||
                range_error[lane] !== invalid[lane])
                $fatal(1, "LANES=%0d lane=%0d mode=%0d shift=%0d check=%0d", LANES, lane, mode, s, checks);
        end
    endtask

    initial begin
        logic [31:0] config_word;
        done = 0;
        for (int s = 0; s <= 31; s++)
            for (int r = 0; r <= 1; r++)
                for (int mode = 0; mode <= 11; mode++) check(mode, s, r);
        repeat (20000) begin
            config_word = random_word();
            check(config_word[6] ? 10 : 11, config_word[4:0], config_word[5]);
        end
        check(0, 1, 0); // Recovery after errors, no latched state.
        $display("PASS: bias output LANES=%0d checks=%0d seed=%h", LANES, checks, 32'h74c0ffee ^ LANES);
        done = 1;
    end
endmodule

module tb_bias_output;
    wire [2:0] done;
    bias_output_case #(.LANES(1)) one_lane(done[0]);
    bias_output_case #(.LANES(3)) three_lanes(done[1]);
    bias_output_case #(.LANES(8)) eight_lanes(done[2]);
    initial begin wait (&done); $display("PASS: bias output all configurations"); $finish; end
    initial begin #1000000; $fatal(1, "Bias output timeout"); end
endmodule
