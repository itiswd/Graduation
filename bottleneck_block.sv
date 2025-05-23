// MobileNetV3 Bottleneck Block (Inverted Residual Block)
// Includes expansion, depthwise convolution, squeeze-excitation, and projection

module bottleneck_block #(
    parameter KERNEL_SIZE = 3,
    parameter IN_CHANNELS = 16,
    parameter EXPANDED_CHANNELS = 72,
    parameter OUT_CHANNELS = 24,
    parameter STRIDE = 1,
    parameter USE_SE = 1,  // 1 to use Squeeze-Excitation, 0 to skip
    parameter USE_HS = 1,  // 1 to use hard-swish, 0 to use ReLU
    parameter INPUT_WIDTH = 112,
    parameter INPUT_HEIGHT = 112,
    parameter FIXED_WIDTH = 16,
    parameter FRAC_WIDTH = 8,
    parameter PARALLEL_MACS = 16
)(
    input logic clk,
    input logic rst_n,
    input logic start,
    input logic signed [FIXED_WIDTH-1:0] input_data [IN_CHANNELS-1:0][INPUT_HEIGHT-1:0][INPUT_WIDTH-1:0],
    output logic signed [FIXED_WIDTH-1:0] output_data [OUT_CHANNELS-1:0][(INPUT_HEIGHT-1)/STRIDE:0][(INPUT_WIDTH-1)/STRIDE:0],
    output logic valid_out,
    output logic ready
);

    localparam OUTPUT_WIDTH = (INPUT_WIDTH - 1) / STRIDE + 1;
    localparam OUTPUT_HEIGHT = (INPUT_HEIGHT - 1) / STRIDE + 1;
    localparam USE_RESIDUAL = (STRIDE == 1 && IN_CHANNELS == OUT_CHANNELS);

    // Pipeline stage outputs
    logic signed [FIXED_WIDTH-1:0] expansion_out [EXPANDED_CHANNELS-1:0][INPUT_HEIGHT-1:0][INPUT_WIDTH-1:0];
    logic signed [FIXED_WIDTH-1:0] depthwise_out [EXPANDED_CHANNELS-1:0][OUTPUT_HEIGHT-1:0][OUTPUT_WIDTH-1:0];
    logic signed [FIXED_WIDTH-1:0] se_out [EXPANDED_CHANNELS-1:0][OUTPUT_HEIGHT-1:0][OUTPUT_WIDTH-1:0];
    logic signed [FIXED_WIDTH-1:0] projection_out [OUT_CHANNELS-1:0][OUTPUT_HEIGHT-1:0][OUTPUT_WIDTH-1:0];

    // Control signals
    logic expansion_valid, depthwise_valid, se_valid, projection_valid;
    logic expansion_ready, depthwise_ready, se_ready, projection_ready;

    // Skip expansion if input channels equal expanded channels
    generate
        if (IN_CHANNELS != EXPANDED_CHANNELS) begin : expansion_layer
            // 1x1 Pointwise Expansion
            conv2d_layer #(
                .IN_CHANNELS(IN_CHANNELS),
                .OUT_CHANNELS(EXPANDED_CHANNELS),
                .KERNEL_SIZE(1),
                .STRIDE(1),
                .PADDING(0),
                .INPUT_WIDTH(INPUT_WIDTH),
                .INPUT_HEIGHT(INPUT_HEIGHT),
                .PARALLEL_MACS(PARALLEL_MACS),
                .FIXED_WIDTH(FIXED_WIDTH),
                .FRAC_WIDTH(FRAC_WIDTH)
            ) expansion_conv (
                .clk(clk),
                .rst_n(rst_n),
                .start(start),
                .input_data(input_data),
                .weights(), // Weights would be loaded from memory
                .bias(),    // Bias would be loaded from memory
                .output_data(expansion_out),
                .valid_out(expansion_valid),
                .ready(expansion_ready)
            );

            // Batch Norm + Activation after expansion
            genvar i, j, k;
            if (USE_HS) begin : expansion_hswish
                for (i = 0; i < EXPANDED_CHANNELS; i++) begin : exp_hs_ch
                    for (j = 0; j < INPUT_HEIGHT; j++) begin : exp_hs_h
                        for (k = 0; k < INPUT_WIDTH; k++) begin : exp_hs_w
                            hswish #(
                                .FIXED_WIDTH(FIXED_WIDTH),
                                .FRAC_WIDTH(FRAC_WIDTH)
                            ) hswish_inst (
                                .clk(clk),
                                .rst_n(rst_n),
                                .x(expansion_out[i][j][k]),
                                .y(expansion_out[i][j][k]) // In-place activation
                            );
                        end
                    end
                end
            end else begin : expansion_relu
                for (i = 0; i < EXPANDED_CHANNELS; i++) begin : exp_relu_ch
                    for (j = 0; j < INPUT_HEIGHT; j++) begin : exp_relu_h
                        for (k = 0; k < INPUT_WIDTH; k++) begin : exp_relu_w
                            relu #(
                                .FIXED_WIDTH(FIXED_WIDTH),
                                .FRAC_WIDTH(FRAC_WIDTH)
                            ) relu_inst (
                                .x(expansion_out[i][j][k]),
                                .y(expansion_out[i][j][k]) // In-place activation
                            );
                        end
                    end
                end
            end
        end else begin : no_expansion
            assign expansion_out = input_data;
            assign expansion_valid = start;
            assign expansion_ready = 1'b1;
        end
    endgenerate

    // Depthwise Convolution
    depthwise_conv2d #(
        .CHANNELS(EXPANDED_CHANNELS),
        .KERNEL_SIZE(KERNEL_SIZE),
        .STRIDE(STRIDE),
        .PADDING((KERNEL_SIZE-1)/2),
        .INPUT_WIDTH(INPUT_WIDTH),
        .INPUT_HEIGHT(INPUT_HEIGHT),
        .FIXED_WIDTH(FIXED_WIDTH),
        .FRAC_WIDTH(FRAC_WIDTH)
    ) depthwise_conv (
        .clk(clk),
        .rst_n(rst_n),
        .start(expansion_valid),
        .input_data(expansion_out),
        .weights(), // Depthwise weights from memory
        .bias(),    // Depthwise bias from memory
        .output_data(depthwise_out),
        .valid_out(depthwise_valid),
        .ready(depthwise_ready)
    );

    // Squeeze-and-Excitation (optional)
    generate
        if (USE_SE) begin : se_module
            squeeze_excitation #(
                .CHANNELS(EXPANDED_CHANNELS),
                .REDUCTION_RATIO(4),
                .WIDTH(OUTPUT_WIDTH),
                .HEIGHT(OUTPUT_HEIGHT),
                .FIXED_WIDTH(FIXED_WIDTH),
                .FRAC_WIDTH(FRAC_WIDTH)
            ) se_inst (
                .clk(clk),
                .rst_n(rst_n),
                .start(depthwise_valid),
                .input_data(depthwise_out),
                .output_data(se_out),
                .valid_out(se_valid),
                .ready(se_ready)
            );
        end else begin : no_se
            assign se_out = depthwise_out;
            assign se_valid = depthwise_valid;
            assign se_ready = depthwise_ready;
        end
    endgenerate

    // 1x1 Pointwise Projection (no activation)
    conv2d_layer #(
        .IN_CHANNELS(EXPANDED_CHANNELS),
        .OUT_CHANNELS(OUT_CHANNELS),
        .KERNEL_SIZE(1),
        .STRIDE(1),
        .PADDING(0),
        .INPUT_WIDTH(OUTPUT_WIDTH),
        .INPUT_HEIGHT(OUTPUT_HEIGHT),
        .PARALLEL_MACS(PARALLEL_MACS),
        .FIXED_WIDTH(FIXED_WIDTH),
        .FRAC_WIDTH(FRAC_WIDTH)
    ) projection_conv (
        .clk(clk),
        .rst_n(rst_n),
        .start(se_valid),
        .input_data(se_out),
        .weights(), // Projection weights from memory
        .bias(),    // Projection bias from memory
        .output_data(projection_out),
        .valid_out(projection_valid),
        .ready(projection_ready)
    );

    // Residual connection (if applicable)
    generate
        if (USE_RESIDUAL) begin : residual_connection
            genvar i, j, k;
            for (i = 0; i < OUT_CHANNELS; i++) begin : res_ch
                for (j = 0; j < OUTPUT_HEIGHT; j++) begin : res_h
                    for (k = 0; k < OUTPUT_WIDTH; k++) begin : res_w
                        always_ff @(posedge clk or negedge rst_n) begin
                            if (!rst_n) begin
                                output_data[i][j][k] <= '0;
                            end else if (projection_valid) begin
                                // Add residual connection
                                output_data[i][j][k] <= projection_out[i][j][k] + input_data[i][j][k];
                            end
                        end
                    end
                end
            end
        end else begin : no_residual
            assign output_data = projection_out;
        end
    endgenerate

    assign valid_out = projection_valid;
    assign ready = projection_ready;

endmodule

// Depthwise Convolution Module
module depthwise_conv2d #(
    parameter CHANNELS = 72,
    parameter KERNEL_SIZE = 3,
    parameter STRIDE = 1,
    parameter PADDING = 1,
    parameter INPUT_WIDTH = 112,
    parameter INPUT_HEIGHT = 112,
    parameter FIXED_WIDTH = 16,
    parameter FRAC_WIDTH = 8
)(
    input logic clk,
    input logic rst_n,
    input logic start,
    input logic signed [FIXED_WIDTH-1:0] input_data [CHANNELS-1:0][INPUT_HEIGHT-1:0][INPUT_WIDTH-1:0],
    input logic signed [FIXED_WIDTH-1:0] weights [CHANNELS-1:0][KERNEL_SIZE-1:0][KERNEL_SIZE-1:0],
    input logic signed [FIXED_WIDTH-1:0] bias [CHANNELS-1:0],
    output logic signed [FIXED_WIDTH-1:0] output_data [CHANNELS-1:0][(INPUT_HEIGHT+2*PADDING-KERNEL_SIZE)/STRIDE:0][(INPUT_WIDTH+2*PADDING-KERNEL_SIZE)/STRIDE:0],
    output logic valid_out,
    output logic ready
);

    localparam OUTPUT_WIDTH = (INPUT_WIDTH + 2*PADDING - KERNEL_SIZE) / STRIDE + 1;
    localparam OUTPUT_HEIGHT = (INPUT_HEIGHT + 2*PADDING - KERNEL_SIZE) / STRIDE + 1;

    // Padded input
    logic signed [FIXED_WIDTH-1:0] padded_input [CHANNELS-1:0][INPUT_HEIGHT+2*PADDING-1:0][INPUT_WIDTH+2*PADDING-1:0];

    // Control signals
    logic [7:0] channel_cnt;
    logic [7:0] out_row_cnt, out_col_cnt;
    logic [2:0] kernel_row_cnt, kernel_col_cnt;
    logic [2:0] pipeline_stage;

    typedef enum logic [1:0] {
        IDLE,
        COMPUTE,
        OUTPUT,
        DONE
    } state_t;
    
    state_t current_state, next_state;

    // Add padding
    genvar c, h, w;
    generate
        for (c = 0; c < CHANNELS; c++) begin : pad_channels
            for (h = 0; h < INPUT_HEIGHT + 2*PADDING; h++) begin : pad_height
                for (w = 0; w < INPUT_WIDTH + 2*PADDING; w++) begin : pad_width
                    always_comb begin
                        if (h < PADDING || h >= INPUT_HEIGHT + PADDING || 
                            w < PADDING || w >= INPUT_WIDTH + PADDING) begin
                            padded_input[c][h][w] = '0;
                        end else begin
                            padded_input[c][h][w] = input_data[c][h-PADDING][w-PADDING];
                        end
                    end
                end
            end
        end
    endgenerate

    // Depthwise convolution computation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
            channel_cnt <= '0;
            out_row_cnt <= '0;
            out_col_cnt <= '0;
            kernel_row_cnt <= '0;
            kernel_col_cnt <= '0;
            pipeline_stage <= '0;
            valid_out <= 1'b0;
            ready <= 1'b1;
        end else begin
            current_state <= next_state;
            
            case (current_state)
                IDLE: begin
                    if (start) begin
                        ready <= 1'b0;
                        valid_out <= 1'b0;
                        channel_cnt <= '0;
                        out_row_cnt <= '0;
                        out_col_cnt <= '0;
                        kernel_row_cnt <= '0;
                        kernel_col_cnt <= '0;
                        pipeline_stage <= '0;
                    end
                end
                
                COMPUTE: begin
                    pipeline_stage <= pipeline_stage + 1;
                    
                    if (pipeline_stage >= 2) begin
                        // Perform depthwise convolution for current channel
                        logic signed [FIXED_WIDTH*2-1:0] conv_sum;
                        conv_sum = '0;
                        
                        for (int kr = 0; kr < KERNEL_SIZE; kr++) begin
                            for (int kc = 0; kc < KERNEL_SIZE; kc++) begin
                                conv_sum = conv_sum + 
                                    padded_input[channel_cnt][out_row_cnt*STRIDE + kr][out_col_cnt*STRIDE + kc] *
                                    weights[channel_cnt][kr][kc];
                            end
                        end
                        
                        // Add bias and store result
                        output_data[channel_cnt][out_row_cnt][out_col_cnt] = 
                            conv_sum[FIXED_WIDTH*2-1:FRAC_WIDTH] + bias[channel_cnt];
                    end
                    
                    // Update counters
                    if (out_col_cnt == OUTPUT_WIDTH - 1) begin
                        out_col_cnt <= '0;
                        if (out_row_cnt == OUTPUT_HEIGHT - 1) begin
                            out_row_cnt <= '0;
                            if (channel_cnt == CHANNELS - 1) begin
                                channel_cnt <= '0;
                            end else begin
                                channel_cnt <= channel_cnt + 1;
                            end
                        end else begin
                            out_row_cnt <= out_row_cnt + 1;
                        end
                    end else begin
                        out_col_cnt <= out_col_cnt + 1;
                    end
                end
                
                OUTPUT: begin
                    valid_out <= 1'b1;
                    ready <= 1'b1;
                end
            endcase
        end
    end

    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            IDLE: begin
                if (start) next_state = COMPUTE;
            end
            
            COMPUTE: begin
                if (channel_cnt == CHANNELS - 1 && 
                    out_row_cnt == OUTPUT_HEIGHT - 1 && 
                    out_col_cnt == OUTPUT_WIDTH - 1 &&
                    pipeline_stage >= 3) begin
                    next_state = OUTPUT;
                end
            end
            
            OUTPUT: begin
                next_state = IDLE;
            end
        endcase
    end

endmodule

// Bottleneck Sequence Module (all 11 blocks from MobileNetV3-Small)
module bottleneck_sequence (
    input logic clk,
    input logic rst_n,
    input logic start,
    input logic signed [15:0] input_data [15:0][111:0][111:0],
    output logic signed [15:0] output_data [95:0][6:0][6:0],
    output logic valid_out,
    output logic ready
);

    // Intermediate signals between blocks (corrected dimensions)
    logic signed [15:0] block1_out [23:0][55:0][55:0];    // 24 channels, 56x56
    logic signed [15:0] block2_out [23:0][55:0][55:0];    // 24 channels, 56x56
    logic signed [15:0] block3_out [39:0][27:0][27:0];    // 40 channels, 28x28
    logic signed [15:0] block4_out [39:0][27:0][27:0];    // 40 channels, 28x28
    logic signed [15:0] block5_out [39:0][27:0][27:0];    // 40 channels, 28x28
    logic signed [15:0] block6_out [47:0][27:0][27:0];    // 48 channels, 28x28
    logic signed [15:0] block7_out [47:0][27:0][27:0];    // 48 channels, 28x28
    logic signed [15:0] block8_out [95:0][13:0][13:0];    // 96 channels, 14x14
    logic signed [15:0] block9_out [95:0][13:0][13:0];    // 96 channels, 14x14
    logic signed [15:0] block10_out [95:0][13:0][13:0];   // 96 channels, 14x14

    logic block1_valid, block2_valid, block3_valid, block4_valid, block5_valid;
    logic block6_valid, block7_valid, block8_valid, block9_valid, block10_valid, block11_valid;

    // Block 1: (3, 16, 72, 24, ReLU, None, 2) - 16->24 channels, stride 2
    bottleneck_block #(
        .KERNEL_SIZE(3), .IN_CHANNELS(16), .EXPANDED_CHANNELS(72), .OUT_CHANNELS(24), 
        .STRIDE(2), .USE_SE(0), .USE_HS(0), .INPUT_WIDTH(112), .INPUT_HEIGHT(112)
    ) block1 (
        .clk(clk), .rst_n(rst_n), .start(start), .input_data(input_data), 
        .output_data(block1_out), .valid_out(block1_valid), .ready()
    );

    // Block 2: (3, 24, 88, 24, ReLU, None, 1) - 24->24 channels, stride 1
    bottleneck_block #(
        .KERNEL_SIZE(3), .IN_CHANNELS(24), .EXPANDED_CHANNELS(88), .OUT_CHANNELS(24), 
        .STRIDE(1), .USE_SE(0), .USE_HS(0), .INPUT_WIDTH(56), .INPUT_HEIGHT(56)
    ) block2 (
        .clk(clk), .rst_n(rst_n), .start(block1_valid), .input_data(block1_out), 
        .output_data(block2_out), .valid_out(block2_valid), .ready()
    );

    // Block 3: (5, 24, 96, 40, hswish, SE, 2) - 24->40 channels, stride 2, kernel 5x5
    bottleneck_block #(
        .KERNEL_SIZE(5), .IN_CHANNELS(24), .EXPANDED_CHANNELS(96), .OUT_CHANNELS(40), 
        .STRIDE(2), .USE_SE(1), .USE_HS(1), .INPUT_WIDTH(56), .INPUT_HEIGHT(56)
    ) block3 (
        .clk(clk), .rst_n(rst_n), .start(block2_valid), .input_data(block2_out), 
        .output_data(block3_out), .valid_out(block3_valid), .ready()
    );

    // Block 4: (5, 40, 240, 40, hswish, SE, 1) - 40->40 channels, stride 1, kernel 5x5
    bottleneck_block #(
        .KERNEL_SIZE(5), .IN_CHANNELS(40), .EXPANDED_CHANNELS(240), .OUT_CHANNELS(40), 
        .STRIDE(1), .USE_SE(1), .USE_HS(1), .INPUT_WIDTH(28), .INPUT_HEIGHT(28)
    ) block4 (
        .clk(clk), .rst_n(rst_n), .start(block3_valid), .input_data(block3_out), 
        .output_data(block4_out), .valid_out(block4_valid), .ready()
    );

    // Block 5: (5, 40, 240, 40, hswish, SE, 1) - 40->40 channels, stride 1, kernel 5x5
    bottleneck_block #(
        .KERNEL_SIZE(5), .IN_CHANNELS(40), .EXPANDED_CHANNELS(240), .OUT_CHANNELS(40), 
        .STRIDE(1), .USE_SE(1), .USE_HS(1), .INPUT_WIDTH(28), .INPUT_HEIGHT(28)
    ) block5 (
        .clk(clk), .rst_n(rst_n), .start(block4_valid), .input_data(block4_out), 
        .output_data(block5_out), .valid_out(block5_valid), .ready()
    );

    // Block 6: (5, 40, 120, 48, hswish, SE, 1) - 40->48 channels, stride 1, kernel 5x5
    bottleneck_block #(
        .KERNEL_SIZE(5), .IN_CHANNELS(40), .EXPANDED_CHANNELS(120), .OUT_CHANNELS(48), 
        .STRIDE(1), .USE_SE(1), .USE_HS(1), .INPUT_WIDTH(28), .INPUT_HEIGHT(28)
    ) block6 (
        .clk(clk), .rst_n(rst_n), .start(block5_valid), .input_data(block5_out), 
        .output_data(block6_out), .valid_out(block6_valid), .ready()
    );

    // Block 7: (5, 48, 144, 48, hswish, SE, 1) - 48->48 channels, stride 1, kernel 5x5
    bottleneck_block #(
        .KERNEL_SIZE(5), .IN_CHANNELS(48), .EXPANDED_CHANNELS(144), .OUT_CHANNELS(48), 
        .STRIDE(1), .USE_SE(1), .USE_HS(1), .INPUT_WIDTH(28), .INPUT_HEIGHT(28)
    ) block7 (
        .clk(clk), .rst_n(rst_n), .start(block6_valid), .input_data(block6_out), 
        .output_data(block7_out), .valid_out(block7_valid), .ready()
    );

    // Block 8: (5, 48, 288, 96, hswish, SE, 2) - 48->96 channels, stride 2, kernel 5x5
    bottleneck_block #(
        .KERNEL_SIZE(5), .IN_CHANNELS(48), .EXPANDED_CHANNELS(288), .OUT_CHANNELS(96), 
        .STRIDE(2), .USE_SE(1), .USE_HS(1), .INPUT_WIDTH(28), .INPUT_HEIGHT(28)
    ) block8 (
        .clk(clk), .rst_n(rst_n), .start(block7_valid), .input_data(block7_out), 
        .output_data(block8_out), .valid_out(block8_valid), .ready()
    );

    // Block 9: (5, 96, 576, 96, hswish, SE, 1) - 96->96 channels, stride 1, kernel 5x5
    bottleneck_block #(
        .KERNEL_SIZE(5), .IN_CHANNELS(96), .EXPANDED_CHANNELS(576), .OUT_CHANNELS(96), 
        .STRIDE(1), .USE_SE(1), .USE_HS(1), .INPUT_WIDTH(14), .INPUT_HEIGHT(14)
    ) block9 (
        .clk(clk), .rst_n(rst_n), .start(block8_valid), .input_data(block8_out), 
        .output_data(block9_out), .valid_out(block9_valid), .ready()
    );

    // Block 10: (5, 96, 576, 96, hswish, SE, 1) - 96->96 channels, stride 1, kernel 5x5
    bottleneck_block #(
        .KERNEL_SIZE(5), .IN_CHANNELS(96), .EXPANDED_CHANNELS(576), .OUT_CHANNELS(96), 
        .STRIDE(1), .USE_SE(1), .USE_HS(1), .INPUT_WIDTH(14), .INPUT_HEIGHT(14)
    ) block10 (
        .clk(clk), .rst_n(rst_n), .start(block9_valid), .input_data(block9_out), 
        .output_data(block10_out), .valid_out(block10_valid), .ready()
    );

    // Block 11: Final block with stride 2 to downsample from 14x14 to 7x7
    // (5, 96, 576, 96, hswish, SE, 2) - 96->96 channels, stride 2, kernel 5x5
    bottleneck_block #(
        .KERNEL_SIZE(5), .IN_CHANNELS(96), .EXPANDED_CHANNELS(576), .OUT_CHANNELS(96), 
        .STRIDE(2), .USE_SE(1), .USE_HS(1), .INPUT_WIDTH(14), .INPUT_HEIGHT(14)
    ) block11 (
        .clk(clk), .rst_n(rst_n), .start(block10_valid), .input_data(block10_out), 
        .output_data(output_data), .valid_out(block11_valid), .ready()
    );

    assign valid_out = block11_valid;
    assign ready = 1'b1; // Always ready for new data when not processing

endmodule 