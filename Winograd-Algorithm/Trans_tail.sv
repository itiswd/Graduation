//trans_tail_pipelined_fixed_no_immediate
module Trans_tail#(
    parameter width = 16 , parameter rows = 4 , parameter cols = 4
) (clk,rst_n,data_in,I);
    //inputs ports
    input clk,rst_n;
    //input signed [width - 1:0] kernel [0:rows-2][0:cols-2];
    input signed [width - 1:0] data_in  [0: rows-1][0: cols-1];
    //outputs ports
    output logic signed [width - 1:0] I [0: rows -1][0: cols-1];
    // Transformation Matrices B and B_T
        localparam  logic signed [width-1:0] B_T [0:rows-1][0:cols-1] = '{
            '{1,0,-1,0},
            '{0,1,1,0},
            '{0,-1,1,0},
            '{0,1,0,-1}
            };
        localparam  logic signed [width-1:0] B [0:rows-1][0:cols-1] = '{
            '{1,0,0,0},
            '{0,1,-1,1},
            '{-1,1,1,0},
            '{0,0,0,-1}
            };
   
    //Internals signals
    logic signed [width-1:0] D_T [0:rows-1][0:cols-1] ;
    //logic signed [width-1:0] I [0:rows-1][0:cols-1] ;
    //Always Block For The Input tranformation
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i=0; i<rows; i++) begin
                for (int j=0; j<cols; j++) begin 
                    D_T[i][j] <= 0;
                    I[i][j] <= 0;
                end
            end
        end else begin
            for (int i=0; i<rows; i++) begin
                for (int j=0; j<cols; j++) begin
                    D_T[i][j] <= B_T[i][0] * data_in[0][j] + B_T[i][1] * data_in[1][j] + B_T[i][2] * data_in[2][j] + B_T[i][3] * data_in[3][j];
                end
            end
            for (int i=0; i<rows; i++) begin
                for (int j=0; j<cols; j++) begin
                    I[i][j] <= D_T[i][0] * B[0][j] + D_T[i][1] * B[1][j] + D_T[i][2] * B[2][j] + D_T[i][3] * B[3][j];
                end
            end
        end
    end
endmodule