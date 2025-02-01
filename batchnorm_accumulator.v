module batchnorm_accumulator #(parameter WIDTH=16, FRAC=8, BATCH_SIZE=10) (
    input clk,
    input rst,
    input [WIDTH-1:0] x_in, 
    output reg [WIDTH-1:0] mean,  
    output reg [WIDTH-1:0] var    
);

    reg [WIDTH-1:0] sum = 0;     
    reg [WIDTH-1:0] sum_sq = 0;    
    reg [3:0] count = 0;   
            
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            sum <= 0;
            sum_sq <= 0;
            count <= 0;
            mean <= 0;
            var <= 0;
        end else begin
            if (count < BATCH_SIZE) begin
                sum <= sum + x_in;

                sum_sq <= sum_sq + (x_in * x_in);

                count <= count + 1;
            end else begin
                mean <= sum / BATCH_SIZE;

                var <= (sum_sq / BATCH_SIZE) - (mean * mean);

                sum <= 0;
                sum_sq <= 0;
                count <= 0;
            end
        end
    end

endmodule