`timescale 1ns / 1ps
module mac (
    input  logic clk,
    input  logic reset,
    input  logic clear,
    input  logic enable,
    input  logic signed [7:0] a,
    input  logic signed [7:0] b,
    output logic signed [31:0] acc
);
    logic signed [15:0] product;
    logic signed [31:0] extended_product;

    assign product = a * b;
    assign extended_product = {{16{product[15]}}, product};

    always_ff @(posedge clk) begin
        if (reset)
            acc <= 32'sd0;
        else if (clear)
            acc <= 32'sd0;
        else if (enable)
            acc <= acc + extended_product;
    end
endmodule
