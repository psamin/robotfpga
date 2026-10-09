`timescale 1ns / 1ps
module mac_array_case #(parameter int LANES = 8)(output bit done);
    logic clk = 0, reset = 0, clear = 0;
    logic [LANES-1:0] enable = '0;
    logic [LANES-1:0][7:0] a = '0, b = '0;
    wire [LANES-1:0][31:0] acc;
    longint signed expected [LANES];
    int checks = 0;
    logic [31:0] rng = 32'h61c0ffee ^ LANES;
    mac_array #(.LANES(LANES)) dut (.*);
    always #5 clk = ~clk;

    function automatic logic [31:0] random_word();
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return rng;
    endfunction

    task automatic step(input int mode, input bit rst, clr,
                        input logic [LANES-1:0] mask);
        longint signed av, bv;
        logic [31:0] ra, rb;
        @(negedge clk);
        reset = rst; clear = clr; enable = mask;
        for (int lane = 0; lane < LANES; lane++) begin
            case (mode)
                0: begin // Distinct magnitudes and signs expose lane swaps.
                    av = (lane % 2) ? -(lane + 1) : lane + 1;
                    bv = (lane % 3) ? -7 : 13;
                end
                1: begin
                    ra = random_word(); rb = random_word();
                    av = int'(ra[7:0]) - 128; bv = int'(rb[7:0]) - 128;
                end
                2: begin av = -128; bv = -128; end
                3: begin av = -128; bv = 127; end
                default: begin av = 127; bv = 127; end
            endcase
            a[lane] = av; b[lane] = bv;
            if (rst || clr) expected[lane] = 0;
            else if (mask[lane]) expected[lane] += av * bv;
            if (expected[lane] > 64'sd2147483647)
                expected[lane] -= 64'sd4294967296;
            if (expected[lane] < -64'sd2147483648)
                expected[lane] += 64'sd4294967296;
        end
        @(posedge clk); #1;
        for (int lane = 0; lane < LANES; lane++) begin
            checks++;
            if ($signed(acc[lane]) !== expected[lane])
                $fatal(1, "LANES=%0d lane=%0d check=%0d got=%0d expected=%0d",
                       LANES, lane, checks, $signed(acc[lane]), expected[lane]);
        end
    endtask

    initial begin
        logic [31:0] controls;
        logic [LANES-1:0] mask;
        done = 0;
        foreach (expected[lane]) expected[lane] = 0;
        step(0, 1, 0, '0);
        repeat (4) step(0, 0, 0, '1);
        // Only one lane accumulates; all other operands keep changing.
        for (int lane = 0; lane < LANES; lane++)
            repeat (3) step(1, 0, 0, LANES'(1) << lane);
        step(1, 0, 0, '0);
        step(0, 0, 1, '1);
        step(0, 0, 0, '1);
        step(1, 1, 1, '1);
        step(0, 0, 0, '1);
        step(1, 1, 0, '1);
        step(0, 0, 0, '1);
        step(1, 0, 1, '0);
        step(2, 0, 0, '1);
        step(3, 0, 0, '1);
        step(4, 0, 0, '1);
        step(0, 0, 1, '0);
        repeat (131073) step(2, 0, 0, '1);
        step(0, 0, 1, '0);
        repeat (132105) step(3, 0, 0, '1);
        step(0, 0, 1, '0);
        repeat (10000) begin
            controls = random_word();
            for (int lane = 0; lane < LANES; lane++) begin
                controls = random_word(); mask[lane] = controls[0];
            end
            step(1, controls[5:0] == 0, controls[9:6] == 0, mask);
        end
        $display("PASS: MAC array LANES=%0d checks=%0d seed=%h",
                 LANES, checks, 32'h61c0ffee ^ LANES);
        done = 1;
    end
endmodule

module tb_mac_array;
    wire [2:0] done;
    mac_array_case #(.LANES(1)) one_lane(done[0]);
    mac_array_case #(.LANES(3)) three_lanes(done[1]);
    mac_array_case #(.LANES(8)) eight_lanes(done[2]);
    initial begin
        wait (&done);
        $display("PASS: MAC array all configurations");
        $finish;
    end
    initial begin
        #10000000;
        $fatal(1, "MAC array timeout");
    end
endmodule
