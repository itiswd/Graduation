//trans_filter_pipelined_fixed_no_i
module Trans_filter #(
    parameter WIDTH = 16,
    parameter ROWS = 4,
    parameter COLS = 4,
    parameter FRAC_WIDTH = 8
) (
    input clk,
    input rst_n,
    input enable,
    input signed [WIDTH-1:0] kernel [0:ROWS-2][0:COLS-2],  // Assumed Q8.8
    output logic signed [WIDTH-1:0] F [0:ROWS-1][0:COLS-1]  // Output in Q8.8
);

    // Q8.8 fixed-point format: 8 integer bits, 8 fractional bits
    localparam INT_WIDTH = WIDTH - FRAC_WIDTH;

    // G matrix (4x3) in Q8.8 format
    localparam logic signed [WIDTH-1:0] G[0:3][0:2] = '{
        '{16'h0100, 16'h0000, 16'h0000},  // [1, 0, 0]
        '{16'h0080, 16'h0080, 16'h0080},  // [0.5, 0.5, 0.5]
        '{16'h0080, 16'hFF80, 16'h0080},  // [0.5, -0.5, 0.5]
        '{16'h0000, 16'h0000, 16'h0000}   // [0, 0, 1]
    };
    
    // G^T matrix (3x4) in Q8.8 format
    localparam logic signed [WIDTH-1:0] G_T[0:2][0:3] = '{
        '{16'h0100, 16'h0080, 16'h0080, 16'h0000},  // [1, 0.5, 0.5, 0]
        '{16'h0000, 16'h0080, 16'hFF80, 16'h0000},  // [0, 0.5, -0.5, 0]
        '{16'h0000, 16'h0080, 16'h0080, 16'h0000}   // [0, 0.5, 0.5, 1]
    };

    // Pipeline registers
    logic signed [WIDTH-1:0] W_reg [0:ROWS-1][0:COLS-2];  // Stage 1 output in Q8.8
    logic enable_stage2;                                   // Pipeline control

    // Temporary variables for fixed-point multiplication
    logic signed [2*WIDTH-1:0] temp_stage1 [0:ROWS-1][0:COLS-2];
    logic signed [2*WIDTH-1:0] temp_stage2 [0:ROWS-1][0:COLS-1];

    // Stage 1: G * kernel (4x3 * 3x3 = 4x3)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < ROWS; i++) begin
                for (int j = 0; j < COLS-1; j++) begin
                    W_reg[i][j] <= '0;
                end
            end
            enable_stage2 <= '0;
        end
        else begin
            enable_stage2 <= enable;
            if (enable) begin
                for (int i = 0; i < 4; i++) begin
                    for (int j = 0; j < 3; j++) begin
                        // Accumulate full precision, then shift
                        temp_stage1[i][j] = $signed(G[i][0]) * $signed(kernel[0][j]) + 
                                          $signed(G[i][1]) * $signed(kernel[1][j]) + 
                                          $signed(G[i][2]) * $signed(kernel[2][j]);
                        W_reg[i][j] <= temp_stage1[i][j][WIDTH+FRAC_WIDTH-1:FRAC_WIDTH];
                    end
                end
            end
        end
    end

    // Stage 2: W * G^T (4x3 * 3x4 = 4x4)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < ROWS; i++) begin
                for (int j = 0; j < COLS; j++) begin
                    F[i][j] <= '0;
                end
            end
        end
        else if (enable_stage2) begin
            for (int i = 0; i < 4; i++) begin
                for (int j = 0; j < 4; j++) begin
                    // Accumulate full precision, then shift
                    temp_stage2[i][j] = $signed(W_reg[i][0]) * $signed(G_T[0][j]) + 
                                      $signed(W_reg[i][1]) * $signed(G_T[1][j]) + 
                                      $signed(W_reg[i][2]) * $signed(G_T[2][j]);
                    F[i][j] <= temp_stage2[i][j][WIDTH+FRAC_WIDTH-1:FRAC_WIDTH];
                end
            end
        end
    end

endmodule