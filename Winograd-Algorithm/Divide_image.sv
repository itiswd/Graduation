module Divide_Image #(
    parameter width = 16,
    parameter rows = 224,
    parameter cols = 224
) (
    input clk,               
    input rst, 
    input valid, // valid signal to indicate that the data is ready to be processed            
    input signed [width - 1:0] data_in [0: rows-1][0: cols-1], // 224x224x1 channel
    input signed [width - 1:0] kernel [0:2][0:2], // kernel width = 8 with 3x3x1 

    output logic signed [width - 1:0] data_out_even_even [0: rows/2 -1][0: cols/2 -1], // 111x111x1 channel
    output logic signed [width - 1:0] data_out_even_odd [0: rows/2 -1][0: cols/2 -1], // 111x111x1 channel
    output logic signed [width - 1:0] data_out_odd_even [0: rows/2 -1][0: cols/2 -1], // 111x111x1 channel
    output logic signed [width - 1:0] data_out_odd_odd [0: rows/2 -1][0: cols/2 -1], // 111x111x1 channel
    
    output logic signed [width - 1:0] kernel_even_even [0:2][0:2],
    output logic signed [width - 1:0] kernel_even_odd [0:2][0:2],
    output logic signed [width - 1:0] kernel_odd_even [0:2][0:2],   
    output logic signed [width - 1:0] kernel_odd_odd [0:2][0:2],
    output logic process_done // signal to indicate that the processing is done
);

    typedef enum {
        IDLE,
        PROCESSING,
        DONE
    } state_t;
    
    state_t current_state, next_state;
    
    // Counters for processing
    logic [15:0] row_counter;
    logic [15:0] col_counter;
    
    // Reset and state transition
    always @(posedge clk or negedge rst) begin
        if (!rst) begin
            current_state <= IDLE;
            process_done <= 0;
            row_counter <= 0;
            col_counter <= 0;
            
            // Initialize outputs
            for (int i = 0; i < rows/2; i++) begin
                for (int j = 0; j < cols/2; j++) begin
                    data_out_even_even[i][j] <= 0;
                    data_out_even_odd[i][j] <= 0;
                    data_out_odd_even[i][j] <= 0;
                    data_out_odd_odd[i][j] <= 0;
                end
            end
            
            // Initialize kernels with zeros
            for (int i = 0; i < 3; i++) begin
                for (int j = 0; j < 3; j++) begin
                    kernel_even_even[i][j] <= 0;
                    kernel_even_odd[i][j] <= 0;
                    kernel_odd_even[i][j] <= 0;
                    kernel_odd_odd[i][j] <= 0;
                end
            end
            
        end else begin
            current_state <= next_state;
            
            case (current_state)
                IDLE: begin
                    process_done <= 0;
                    row_counter <= 0;
                    col_counter <= 0;
                end
                
                PROCESSING: begin
                    // Process image data
                    if (row_counter < rows && col_counter < cols) begin
                        if (row_counter[0] == 0 && col_counter[0] == 0) begin
                            data_out_even_even[row_counter/2][col_counter/2] <= data_in[row_counter][col_counter];
                        end
                        if (row_counter[0] == 0 && col_counter[0] == 1) begin
                            data_out_even_odd[row_counter/2][col_counter/2] <= data_in[row_counter][col_counter];
                        end
                        if (row_counter[0] == 1 && col_counter[0] == 0) begin
                            data_out_odd_even[row_counter/2][col_counter/2] <= data_in[row_counter][col_counter];
                        end
                        if (row_counter[0] == 1 && col_counter[0] == 1) begin
                            data_out_odd_odd[row_counter/2][col_counter/2] <= data_in[row_counter][col_counter];
                        end
                        
                        // Move to next column
                        col_counter <= col_counter + 1;
                        if (col_counter == cols-1) begin
                            col_counter <= 0;
                            row_counter <= row_counter + 1;
                        end
                    end
                    
                    // Process kernel (only once)
                    if (row_counter == 0 && col_counter == 0) begin
                        // Kernel_even_even (even rows, even columns)
                        // Even rows (0), even cols (0): kernel[0][0]
                    kernel_even_even[0][0] <= kernel[0][0];
                    kernel_even_even[0][1] <= kernel[0][2];
                    kernel_even_even[0][2] <= 0;
                    kernel_even_even[1][0] <= kernel[2][0];
                    kernel_even_even[1][1] <= kernel[2][2];
                    kernel_even_even[1][2] <= 0;
                    kernel_even_even[2][0] <= 0;
                    kernel_even_even[2][1] <= 0;
                    kernel_even_even[2][2] <= 0;
                    
                    // Even rows (0), odd cols (1): kernel[0][1]
                    kernel_even_odd[0][0] <= kernel[0][1];
                    kernel_even_odd[0][1] <= 0;
                    kernel_even_odd[0][2] <= 0;
                    kernel_even_odd[1][0] <= kernel[2][1];
                    kernel_even_odd[1][1] <= 0;
                    kernel_even_odd[1][2] <= 0;
                    kernel_even_odd[2][0] <= 0;
                    kernel_even_odd[2][1] <= 0;
                    kernel_even_odd[2][2] <= 0;
                    
                    // Odd rows (1), even cols (0): kernel[1][0]
                    kernel_odd_even[0][0] <= kernel[1][0];
                    kernel_odd_even[0][1] <= kernel[1][2];
                    kernel_odd_even[0][2] <= 0;
                    kernel_odd_even[1][0] <= 0;
                    kernel_odd_even[1][1] <= 0;
                    kernel_odd_even[1][2] <= 0;
                    kernel_odd_even[2][0] <= 0;
                    kernel_odd_even[2][1] <= 0;
                    kernel_odd_even[2][2] <= 0;
                    
                    // Odd rows (1), odd cols (1): kernel[1][1]
                    kernel_odd_odd[0][0] <= kernel[1][1];
                    kernel_odd_odd[0][1] <= 0;
                    kernel_odd_odd[0][2] <= 0;
                    kernel_odd_odd[1][0] <= 0;
                    kernel_odd_odd[1][1] <= 0;
                    kernel_odd_odd[1][2] <= 0;
                    kernel_odd_odd[2][0] <= 0;
                    kernel_odd_odd[2][1] <= 0;
                    kernel_odd_odd[2][2] <= 0;
                    end
                end
                
                DONE: begin
                    process_done <= 1;
                end
            endcase
        end
    end

    // Next state logic
    always @(*) begin
        case (current_state)
            IDLE: next_state = valid ? PROCESSING : IDLE;
            PROCESSING: next_state = (row_counter == rows) ? DONE : PROCESSING;
            DONE: next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end

endmodule