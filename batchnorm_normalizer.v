module batchnorm_normalizer #(parameter WIDTH=16, FRAC=8) (
    input clk,
    input rst,
    input [WIDTH-1:0] x_in,  
    input [WIDTH-1:0] gamma, 
    input [WIDTH-1:0] beta,  
    input [WIDTH-1:0] mean,  
    input [WIDTH-1:0] var,   
    output reg [WIDTH-1:0] y_out 
);

    reg [WIDTH-1:0] x_norm;  
    reg [WIDTH-1:0] sqrt_var; 
    reg [WIDTH-1:0] epsilon = 16'h0001; 

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            sqrt_var <= 0;
        end else begin
            sqrt_var <= var + epsilon;
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            x_norm <= 0;
        end else begin
            x_norm <= (x_in - mean) / sqrt_var; 
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            y_out <= 0;
        end else begin
            y_out <= gamma * x_norm + beta; 
        end
    end

endmodule