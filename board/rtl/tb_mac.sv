`timescale 1ns / 1ps
module tb_mac;
    logic clk = 0;
    logic reset = 0, clear = 0, enable = 0;
    logic signed [7:0] a = 0, b = 0;
    logic signed [31:0] acc;
    longint signed expected = 0;
    int checks = 0;
    logic [31:0] rng = 32'h59c0ffee;
    mac dut (.*);
    always #5 clk = ~clk;

    // Independent widened integer arithmetic, reduced modulo 2^32.
    task automatic cycle(input int av, bv, input bit rst, clr, en);
        longint signed wide_a, wide_b;
        @(negedge clk);
        a = av; b = bv; reset = rst; clear = clr; enable = en;
        wide_a = av; wide_b = bv;
        if (rst || clr) expected = 0;
        else if (en) expected = expected + wide_a * wide_b;
        if (expected > 64'sd2147483647) expected = expected - 64'sd4294967296;
        if (expected < -64'sd2147483648) expected = expected + 64'sd4294967296;
        @(posedge clk);
        #1; // Sample after the accumulator's nonblocking assignment.
        checks++;
        if ($signed(acc) !== expected)
            $fatal(1, "check=%0d a=%0d b=%0d rst=%b clr=%b en=%b got=%0d expected=%0d",
                   checks, av, bv, rst, clr, en, acc, expected);
    endtask

    function automatic logic [31:0] next_random();
        rng = rng ^ (rng << 13);
        rng = rng ^ (rng >> 17);
        rng = rng ^ (rng << 5);
        return rng;
    endfunction

    initial begin
        logic [31:0] ra, rb, controls;
        cycle(0, 0, 1, 0, 0);
        cycle(3, 4, 0, 0, 1);
        cycle(-3, 4, 0, 0, 1);
        cycle(3, -4, 0, 0, 1);
        cycle(-3, -4, 0, 0, 1);
        cycle(-128, -128, 0, 0, 1);
        cycle(-128, 127, 0, 0, 1);
        cycle(127, 127, 0, 0, 1);
        cycle(0, -128, 0, 0, 1);
        cycle(127, 127, 0, 0, 0);
        cycle(-128, -128, 0, 1, 1);
        cycle(7, 9, 0, 0, 1);
        cycle(127, 127, 1, 1, 1);
        cycle(7, 9, 0, 0, 1);
        cycle(127, 127, 1, 0, 1);
        cycle(7, 9, 0, 0, 1);
        cycle(0, 0, 0, 1, 0);
        // Verify synchronous reset: assertion between edges cannot change acc.
        cycle(7, 9, 0, 0, 1);
        @(negedge clk); reset = 1;
        #1;
        if (acc !== 32'sd63) $fatal(1, "Reset acted asynchronously");
        cycle(0, 0, 1, 0, 0);
        // All 65,536 signed operand pairs, accumulating without clearing.
        for (int av = -128; av <= 127; av++)
            for (int bv = -128; bv <= 127; bv++)
                cycle(av, bv, 0, 0, 1);
        // Cross both INT32 boundaries using normal input transactions.
        cycle(0, 0, 0, 1, 0);
        repeat (131073) cycle(-128, -128, 0, 0, 1);
        cycle(0, 0, 0, 1, 0);
        repeat (132105) cycle(-128, 127, 0, 0, 1);
        cycle(0, 0, 0, 1, 0);
        repeat (10000) begin
            ra = next_random(); rb = next_random(); controls = next_random();
            cycle(int'(ra[7:0]) - 128, int'(rb[7:0]) - 128,
                  controls[5:0] == 0, controls[9:6] == 0, controls[10]);
        end
        $display("PASS: MAC %0d cycle checks; seed=59c0ffee", checks);
        $finish;
    end
    initial begin
        #10000000;
        $fatal(1, "MAC test timeout");
    end
endmodule
