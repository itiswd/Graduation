`timescale 1ns/1ps
module bottleneck_block_tb;
    // Parameters (can be changed for different block configs)
    localparam KERNEL_SIZE = 5;
    localparam IN_CHANNELS = 8;
    localparam EXPANDED_CHANNELS = 16;
    localparam OUT_CHANNELS = 8;
    localparam STRIDE = 2;
    localparam INPUT_WIDTH = 8;
    localparam INPUT_HEIGHT = 8;
    localparam FIXED_WIDTH = 16;
    localparam FRAC_WIDTH = 8;
    localparam PARALLEL_MACS = 4;
    localparam REDUCTION_RATIO = 4;
    localparam TEST_ITER = 4;

    // Signals
    logic clk;
    logic rst_n;
    logic start;
    logic signed [FIXED_WIDTH-1:0] input_data [IN_CHANNELS-1:0][INPUT_HEIGHT-1:0][INPUT_WIDTH-1:0];
    logic signed [FIXED_WIDTH-1:0] output_data_0 [OUT_CHANNELS-1:0][(INPUT_HEIGHT-1)/STRIDE:0][(INPUT_WIDTH-1)/STRIDE:0]; // ReLU, no SE
    logic signed [FIXED_WIDTH-1:0] output_data_1 [OUT_CHANNELS-1:0][(INPUT_HEIGHT-1)/STRIDE:0][(INPUT_WIDTH-1)/STRIDE:0]; // h-swish, no SE
    logic signed [FIXED_WIDTH-1:0] output_data_2 [OUT_CHANNELS-1:0][(INPUT_HEIGHT-1)/STRIDE:0][(INPUT_WIDTH-1)/STRIDE:0]; // ReLU, SE
    logic signed [FIXED_WIDTH-1:0] output_data_3 [OUT_CHANNELS-1:0][(INPUT_HEIGHT-1)/STRIDE:0][(INPUT_WIDTH-1)/STRIDE:0]; // h-swish, SE
    logic valid_out_0, valid_out_1, valid_out_2, valid_out_3;
    logic ready_0, ready_1, ready_2, ready_3;

    // Testbench variables
    int error_count = 0;
    int test_count = 0;
    int mode;
    logic signed [FIXED_WIDTH-1:0] val;
    logic signed [FIXED_WIDTH-1:0] sample;

    // DUT instances (all parameter values are constants)
    bottleneck_block #(
        .KERNEL_SIZE(KERNEL_SIZE), .IN_CHANNELS(IN_CHANNELS), .EXPANDED_CHANNELS(EXPANDED_CHANNELS),
        .OUT_CHANNELS(OUT_CHANNELS), .STRIDE(STRIDE), .USE_SE(0), .USE_HS(0),
        .INPUT_WIDTH(INPUT_WIDTH), .INPUT_HEIGHT(INPUT_HEIGHT), .FIXED_WIDTH(FIXED_WIDTH),
        .FRAC_WIDTH(FRAC_WIDTH), .PARALLEL_MACS(PARALLEL_MACS)
    ) dut_relu_no_se (
        .clk(clk), .rst_n(rst_n), .start(start), .input_data(input_data),
        .output_data(output_data_0), .valid_out(valid_out_0), .ready(ready_0)
    );
    bottleneck_block #(
        .KERNEL_SIZE(KERNEL_SIZE), .IN_CHANNELS(IN_CHANNELS), .EXPANDED_CHANNELS(EXPANDED_CHANNELS),
        .OUT_CHANNELS(OUT_CHANNELS), .STRIDE(STRIDE), .USE_SE(0), .USE_HS(1),
        .INPUT_WIDTH(INPUT_WIDTH), .INPUT_HEIGHT(INPUT_HEIGHT), .FIXED_WIDTH(FIXED_WIDTH),
        .FRAC_WIDTH(FRAC_WIDTH), .PARALLEL_MACS(PARALLEL_MACS)
    ) dut_hswish_no_se (
        .clk(clk), .rst_n(rst_n), .start(start), .input_data(input_data),
        .output_data(output_data_1), .valid_out(valid_out_1), .ready(ready_1)
    );
    bottleneck_block #(
        .KERNEL_SIZE(KERNEL_SIZE), .IN_CHANNELS(IN_CHANNELS), .EXPANDED_CHANNELS(EXPANDED_CHANNELS),
        .OUT_CHANNELS(OUT_CHANNELS), .STRIDE(STRIDE), .USE_SE(1), .USE_HS(0),
        .INPUT_WIDTH(INPUT_WIDTH), .INPUT_HEIGHT(INPUT_HEIGHT), .FIXED_WIDTH(FIXED_WIDTH),
        .FRAC_WIDTH(FRAC_WIDTH), .PARALLEL_MACS(PARALLEL_MACS)
    ) dut_relu_se (
        .clk(clk), .rst_n(rst_n), .start(start), .input_data(input_data),
        .output_data(output_data_2), .valid_out(valid_out_2), .ready(ready_2)
    );
    bottleneck_block #(
        .KERNEL_SIZE(KERNEL_SIZE), .IN_CHANNELS(IN_CHANNELS), .EXPANDED_CHANNELS(EXPANDED_CHANNELS),
        .OUT_CHANNELS(OUT_CHANNELS), .STRIDE(STRIDE), .USE_SE(1), .USE_HS(1),
        .INPUT_WIDTH(INPUT_WIDTH), .INPUT_HEIGHT(INPUT_HEIGHT), .FIXED_WIDTH(FIXED_WIDTH),
        .FRAC_WIDTH(FRAC_WIDTH), .PARALLEL_MACS(PARALLEL_MACS)
    ) dut_hswish_se (
        .clk(clk), .rst_n(rst_n), .start(start), .input_data(input_data),
        .output_data(output_data_3), .valid_out(valid_out_3), .ready(ready_3)
    );

    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk;

    // Main test process
    initial begin
        $display("\n==== Starting Comprehensive Testbench for bottleneck_block ====");
        for (mode = 0; mode < 4; mode++) begin
            $display("\n--- Test mode %0d: %s, SE=%0d ---", mode, (mode==1||mode==3)?"h-swish":"ReLU", (mode>=2));
            for (int iter = 0; iter < TEST_ITER; iter++) begin
                test_count++;
                // Randomize input
                for (int c = 0; c < IN_CHANNELS; c++)
                    for (int h = 0; h < INPUT_HEIGHT; h++)
                        for (int w = 0; w < INPUT_WIDTH; w++)
                            input_data[c][h][w] = $urandom_range(-128, 127);
                // Reset
                rst_n = 0;
                start = 0;
                #20;
                rst_n = 1;
                #10;
                // Start
                start = 1;
                #10;
                start = 0;
                // Wait for output from the active instance
                case (mode)
                    0: wait(valid_out_0);
                    1: wait(valid_out_1);
                    2: wait(valid_out_2);
                    3: wait(valid_out_3);
                endcase
                #20;
                // Check output for X/Z
                for (int c = 0; c < OUT_CHANNELS; c++)
                    for (int h = 0; h < (INPUT_HEIGHT-1)/STRIDE+1; h++)
                        for (int w = 0; w < (INPUT_WIDTH-1)/STRIDE+1; w++) begin
                            case (mode)
                                0: val = output_data_0[c][h][w];
                                1: val = output_data_1[c][h][w];
                                2: val = output_data_2[c][h][w];
                                3: val = output_data_3[c][h][w];
                            endcase
                            if (val === 'x || val === 'z) begin
                                $display("Error: X/Z at out[%0d][%0d][%0d] in mode %0d, iter %0d", c, h, w, mode, iter);
                                error_count++;
                            end
                        end
                // Print a sample of the output
                case (mode)
                    0: sample = output_data_0[0][0][0];
                    1: sample = output_data_1[0][0][0];
                    2: sample = output_data_2[0][0][0];
                    3: sample = output_data_3[0][0][0];
                endcase
                $display("Sample output (mode %0d, iter %0d): %0d", mode, iter, sample);
                #20;
            end
        end
        $display("\n==== Testbench completed. Total tests: %0d, Errors: %0d ====", test_count, error_count);
        $finish;
    end

    // Monitor
    initial begin
        $monitor("Time=%0t valid_out_0=%b valid_out_1=%b valid_out_2=%b valid_out_3=%b ready_0=%b ready_1=%b ready_2=%b ready_3=%b start=%b", $time, valid_out_0, valid_out_1, valid_out_2, valid_out_3, ready_0, ready_1, ready_2, ready_3, start);
    end
endmodule 