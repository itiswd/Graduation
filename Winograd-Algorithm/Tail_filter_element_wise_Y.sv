module Tail_filter_element_wise_Y #(
    parameter width = 16,
    parameter rows = 224,
    parameter cols = 224
) (
    input clk,
    input rst_n,
    input signed [width-1:0] data_in [0:rows/2-1][0:cols/2-1], // 112x112
    input signed [width-1:0] kernel [0:2][0:2], // 3x3 kernel
    output logic signed [width-1:0] data_out [0:rows/2-1][0:cols/2-1] // 112x112
);

    localparam ROWS_HALF = rows/2; // 112
    localparam COLS_HALF = cols/2; // 112

    // Registers for tail, transformed data, and output
    logic signed [width-1:0] tail [0:3][0:3]; // 4x4 patch
    logic signed [width-1:0] F_kernel [0:3][0:3]; // Transformed kernel
    logic signed [width-1:0] I_tail [0:3][0:3]; // Transformed tail
    logic signed [width-1:0] I_mult_F [0:3][0:3]; // Element-wise multiplication
    logic signed [width-1:0] data_out_temp [0:1][0:1]; // Transformed output

    // Control registers
    logic [6:0] movei, movej; // 7 bits to count up to 112
    logic [1:0] state; // FSM states
    localparam IDLE = 2'd0, LOAD_TAIL = 2'd1, PROCESS = 2'd2, DONE = 2'd3;

    // FSM for controlling the patch processing
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            movei <= 0;
            movej <= 0;
            state <= IDLE;
            for (integer i = 0; i < ROWS_HALF; i++) begin
                for (integer j = 0; j < COLS_HALF; j++) begin
                    data_out[i][j] <= 0;
                end
            end
        end else begin
            case (state)
                IDLE: begin
                    state <= LOAD_TAIL;
                end
                LOAD_TAIL: begin
                    // Load 4x4 tail patch from data_in
                    for (integer i = 0; i < 4; i++) begin
                        for (integer j = 0; j < 4; j++) begin
                            if ((movei + i < ROWS_HALF) && (movej + j < COLS_HALF))
                                tail[i][j] <= data_in[movei + i][movej + j];
                            else
                                tail[i][j] <= 0; // Zero-pad if out of bounds
                        end
                    end
                    state <= PROCESS;
                end
                PROCESS: begin
                    // Write processed patch to data_out
                    for (integer i = 0; i < 4; i++) begin
                        for (integer j = 0; j < 4; j++) begin
                            if ((movei + i < ROWS_HALF) && (movej + j < COLS_HALF))
                                data_out[movei + i][movej + j] <= data_out_temp[i][j];
                        end
                    end
                    // Update movei, movej with stride 2
                    if (movej + 2 >= COLS_HALF) begin
                        movej <= 0;
                        if (movei + 2 >= ROWS_HALF) begin
                            movei <= 0;
                            state <= DONE;
                        end else begin
                            movei <= movei + 2;
                            state <= LOAD_TAIL;
                        end
                    end else begin
                        movej <= movej + 2;
                        state <= LOAD_TAIL;
                    end
                end
                DONE: begin
                    state <= DONE; // Stay in DONE until reset
                end
            endcase
        end
    end

    // Instantiate transformation modules
    Trans_filter Trans_filter_inst (
        .clk(clk),
        .rst_n(rst_n),
        .kernel(kernel),
        .enable(1'b1),
        .F(F_kernel)
    );

    Trans_tail Trans_tail_inst (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(tail),
        .I(I_tail)
    );

    Element_wise_mult Element_wise_inst (
        .clk(clk),
        .rst_n(rst_n),
        .F(F_kernel),
        .I(I_tail),
        .IF(I_mult_F)
    );

    Trans_Y Trans_Y_inst (
        .clk(clk),
        .rst_n(rst_n),
        .IF(I_mult_F),
        .Y(data_out_temp)
    );

endmodule