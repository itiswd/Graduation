
module batchnorm_top #(parameter WIDTH=16, FRAC=8, BATCH_SIZE=10) (
    input clk,
    input rst,
    input [WIDTH-1:0] x_in, 
    input [WIDTH-1:0] gamma, 
    input [WIDTH-1:0] beta,  
    output [WIDTH-1:0] y_out
);

    wire [WIDTH-1:0] mean, var;

    // Accumulate inputs and compute mean/variance
    batchnorm_accumulator #(WIDTH, FRAC, BATCH_SIZE) accumulator (
        .clk(clk),
        .rst(rst),
        .x_in(x_in),
        .mean(mean),
        .var(var)
    );

    batchnorm_normalizer #(WIDTH, FRAC) normalizer (
        .clk(clk),
        .rst(rst),
        .x_in(x_in),
        .gamma(gamma),
        .beta(beta),
        .mean(mean),
        .var(var),
        .y_out(y_out)
    );

endmodule