`timescale 1ns / 1ps

module batchnorm_top_tb;

    parameter WIDTH = 16;
    parameter FRAC = 8;
    parameter BATCH_SIZE = 10;

    reg clk;
    reg rst;
    reg [WIDTH-1:0] x_in;
    reg [WIDTH-1:0] gamma;
    reg [WIDTH-1:0] beta;
    wire [WIDTH-1:0] y_out;

    batchnorm_top #(WIDTH, FRAC, BATCH_SIZE) uut (
        .clk(clk),
        .rst(rst),
        .x_in(x_in),
        .gamma(gamma),
        .beta(beta),
        .y_out(y_out)
    );
integer i ;
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        rst = 1;
        x_in = 0;
        gamma = 16'h0100;
        beta = 16'h0000;
        #10;
        rst = 0;
        for (i= 0; i < BATCH_SIZE; i = i + 1) begin
            x_in = i * 16'h0100;
            #10;
        end
        #100;
        for (i = 0; i < BATCH_SIZE; i = i + 1) begin
            x_in = (i + 10) * 16'h0100;
            #10;
        end
        #100;
        $stop;
    end

    initial begin
        $monitor("Time: %0t | x_in: %h | y_out: %h", $time, x_in, y_out);
    end

endmodule