module Element_wise_mult #(
    parameter WIDTH = 16,
    parameter ROWS = 4,
    parameter COLS = 4,
    parameter FRAC_WIDTH = 8
)(
    input  clk,
    input  rst_n,
    input signed [WIDTH-1:0] I [0: ROWS-1][0: COLS-1],  // First 4x4 matrix (Q8.8)
    input signed [WIDTH-1:0] F [0: ROWS-1][0: COLS-1],  // Second 4x4 matrix (Q8.8)

    output logic signed [WIDTH-1:0] IF [0: ROWS-1][0: COLS-1]   // IF matrix (Q8.8)
);

    // Internal registers for pipelined multiplication
    logic [31:0] product [0:3][0:3];  // Intermediate 32-bit products

    always @(posedge clk or posedge rst_n) begin
        if (rst_n) begin
            // Clear IF matrix
            for (int i = 0; i < ROWS; i = i + 1) begin
                for (int j = 0; j < COLS; j = j + 1) begin
                    IF[i][j] <= 0;
                    product[i][j] <= 0;
                end
            end
        end else begin
            // Perform element-wise multiplication (Q8.8 * Q8.8 = Q16.16)
            for (int i = 0; i < ROWS; i = i + 1) begin
                for (int j = 0; j < COLS; j = j + 1) begin
                    product[i][j] <= $signed(I[i][j]) * $signed(F[i][j]);
                    IF[i][j] <= product[i][j][WIDTH+FRAC_WIDTH-1:FRAC_WIDTH];
                end
            end
            
        end
    end
    
endmodule