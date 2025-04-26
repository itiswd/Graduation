module Trans_Y #(
    parameter width = 16 , parameter rows = 4 , parameter cols = 4, parameter FRAC_WIDTH = 8
) (clk,rst_n,IF,Y);
    //inputs ports
    input clk,rst_n;
    input signed [width - 1:0] IF  [0: rows-1][0: cols-1];
    //outputs ports
    output logic signed [width - 1:0] Y [0: rows/2 -1][0: cols/2-1];
    // Transformation Matrices A and A_T
        localparam  logic signed [width-1:0] A_T [0:1][0:3] = '{
            '{1,1,1,0},
            '{0,1,-1,-1}
            };
        localparam  logic signed [width-1:0] A [0:3][0:1] = '{
            '{1,0},
            '{1,1},
            '{1,-1},
            '{0,-1}
            };
   
    //Internals signals

    logic signed [2*width-1:0] temp_stage1 [0:1][0:3];
    logic signed [width-1:0] IF_T [0:1][0:3] ; 
    logic signed [2*width-1:0] temp_stage2 [0:1][0:1];
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i=0; i<rows/2; i++) begin
                for (int j=0; j<cols/2; j++) begin 
                    temp_stage2[i][j] <= 0;
                    Y[i][j] <= 0;
                end
            end
            for (int i=0; i<rows/2; i++) begin
                for (int j=0; j<cols; j++) begin 
                    temp_stage1[i][j] <= 0;
                    IF_T[i][j] <= 0;
                end
            end
            
        end else begin
            for (int i=0; i<rows/2; i++) begin
                for (int j=0; j<cols; j++) begin
                    temp_stage1[i][j] <= A_T[i][0] * IF[0][j] + 
                                         A_T[i][1] * IF[1][j] + 
                                         A_T[i][2] * IF[2][j] + 
                                         A_T[i][3] * IF[3][j];
                    IF_T[i][j] <= temp_stage1[i][j][width+FRAC_WIDTH-1:FRAC_WIDTH];
                end
            end
            for (int i=0; i<rows/2; i++) begin
                for (int j=0; j<cols/2; j++) begin
                    temp_stage2[i][j] <= IF_T[i][0] * A[0][j] +
                                         IF_T[i][1] * A[1][j] +
                                         IF_T[i][2] * A[2][j] +
                                         IF_T[i][3] * A[3][j];
                    Y[i][j] <= temp_stage2[i][j][width+FRAC_WIDTH-1:FRAC_WIDTH];
                end
            end
        end
    end
endmodule