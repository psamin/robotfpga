`timescale 1ns / 1ps
module tb_requant;
    logic signed [31:0] acc;
    logic [4:0] shift;
    logic relu, bad_shift;
    logic signed [7:0] y;
    int checks = 0;
    logic [31:0] rng = 32'h69c0ffee;
    requant dut (.*);

    function automatic logic [31:0] random_word();
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        return rng;
    endfunction

    task automatic check(input longint signed av, input int s, input bit r);
        longint signed divisor, numerator, expected, lo;
        bit invalid;
        if (av < -64'sd2147483648 || av > 64'sd2147483647)
            $fatal(1, "Test input outside INT32");
        acc = av; shift = s; relu = r;
        invalid = (s < 1 || s > 30);
        expected = 0;
        if (!invalid) begin
            divisor = 64'sd1 << s;
            numerator = av + divisor / 2;
            // SV division truncates toward zero; explicitly compute floor.
            if (numerator < 0) expected = -((-numerator + divisor - 1) / divisor);
            else expected = numerator / divisor;
            lo = r ? 0 : -127;
            if (expected < lo) expected = lo;
            if (expected > 127) expected = 127;
        end
        #1;
        checks++;
        if (bad_shift !== invalid || $signed(y) !== expected)
            $fatal(1, "check=%0d acc=%0d shift=%0d relu=%b y=%0d expected=%0d bad=%b",
                   checks, av, s, r, y, expected, bad_shift);
    endtask

    initial begin
        longint signed divisor, boundary, av_file;
        logic [31:0] word, config_word;
        string vector_path;
        int fd, count, scanned, s_file, r_file, golden, extra;
        for (int s = 1; s <= 30; s++) begin
            divisor = 64'sd1 << s;
            for (int r = 0; r <= 1; r++) begin
                for (int av = -1024; av < 1024; av++) check(av, s, r);
                check(-64'sd2147483648, s, r);
                check(64'sd2147483647, s, r);
                // Halfway ties and their neighbors, including clamp transitions.
                for (int k = -129; k <= 128; k++) begin
                    boundary = k * divisor - divisor / 2;
                    for (int delta = -1; delta <= 1; delta++)
                        if (boundary + delta >= -64'sd2147483648 &&
                            boundary + delta <= 64'sd2147483647)
                            check(boundary + delta, s, r);
                end
            end
        end
        repeat (100000) begin
            word = random_word(); config_word = random_word();
            check($signed(word), 1 + (config_word[15:0] % 30), config_word[16]);
        end
        for (int r = 0; r <= 1; r++) begin
            check(-64'sd2147483648, 0, r); check(64'sd2147483647, 31, r);
            check(0, 0, r); check(0, 31, r);
        end
        // Return to valid operation after error; no latched state.
        check(-3, 1, 0); check(-3, 1, 1); check(-1, 1, 0);
        if ($value$plusargs("VECTORS=%s", vector_path)) begin
            fd = $fopen(vector_path, "r");
            if (!fd) $fatal(1, "Cannot open reference vectors");
            scanned = $fscanf(fd, "%d", count);
            if (scanned != 1 || count < 1) $fatal(1, "Invalid reference count");
            repeat (count) begin
                scanned = $fscanf(fd, "%d %d %d %d", av_file, s_file, r_file, golden);
                if (scanned != 4 || s_file < 1 || s_file > 30 ||
                    r_file < 0 || r_file > 1 || golden < -127 || golden > 127)
                    $fatal(1, "Malformed or truncated reference vectors");
                check(av_file, s_file, r_file);
                if ($signed(y) !== golden) $fatal(1, "Actual NumPy reference mismatch");
            end
            scanned = $fscanf(fd, "%d", extra);
            if (scanned != -1) $fatal(1, "Extra reference data");
            $fclose(fd);
            $display("REFERENCE PASS: %0d actual NumPy cases", count);
        end
        $display("PASS: requant %0d checks; seed=69c0ffee", checks);
        $finish;
    end
    initial begin #1000000; $fatal(1, "Requant test timeout"); end
endmodule
