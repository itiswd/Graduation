module Winograd_TOP #(parameter width = 16 , parameter rows = 224 ,parameter cols = 224) (
    input clk,               
    input rst_n, 
    input valid, // valid signal to indicate that the data is ready to be processed            
    input signed [width - 1:0] data_in  [0: rows-1][0: cols-1], // 224x224x1 channel
    input signed [width - 1:0] kernel [0:2][0:2], // kernel width = 8 with 3x3x1 

    output logic signed [width - 1:0] data_out [0: rows/2 -1][0: cols/2 -1], // 112x112x1 channel
    output logic Convolution_done // signal to indicate that the processing is done
);

    logic signed [width - 1:0] data_out_even_even [0: rows/2 -1][0: cols/2 -1]; // 112x112x1 channel
    logic signed [width - 1:0] data_out_even_odd [0: rows/2 -1][0: cols/2 -1]; // 112x112x1 channel
    logic signed [width - 1:0] data_out_odd_even [0: rows/2 -1][0: cols/2 -1]; // 112x112x1 channel
    logic signed [width - 1:0] data_out_odd_odd [0: rows/2 -1][0: cols/2 -1]; // 112x112x1 channel
    logic signed [width - 1:0] kernel_even_even [0:2][0:2];
    logic signed [width - 1:0] kernel_even_odd [0:2][0:2];
    logic signed [width - 1:0] kernel_odd_even [0:2][0:2];
    logic signed [width - 1:0] kernel_odd_odd [0:2][0:2];
    logic divide_done; // signal to indicate that the processing is done
    
    Divide_image U_Divide_image(
        .clk(clk),
        .rst_n(rst_n),
        .valid(valid), // valid signal to indicate that the data is ready to be processed            
        .data_in(data_in),
        .kernel(kernel), 
        .data_out_even_even(data_out_even_even),
        .data_out_even_odd(data_out_even_odd),
        .data_out_odd_even(data_out_odd_even),
        .data_out_odd_odd(data_out_odd_odd),
        .kernel_even_even(kernel_even_even),
        .kernel_even_odd(kernel_even_odd),
        .kernel_odd_even(kernel_odd_even),   
        .kernel_odd_odd(kernel_odd_odd),
        .divide_done(divide_done) // signal to indicate that the processing is done
    );

    Total_Y U_Total_Y(
        .clk(clk),
        .rst_n(rst_n),
        .divide_done(divide_done),           
        .data_out_even_even(data_out_even_even),
        .data_out_even_odd(data_out_even_odd),
        .data_out_odd_even(data_out_odd_even),
        .data_out_odd_odd(data_out_odd_odd),
        .kernel_even_even(kernel_even_even),
        .kernel_even_odd(kernel_even_odd),
        .kernel_odd_even(kernel_odd_even),   
        .kernel_odd_odd(kernel_odd_odd),
        .data_out(data_out), // 112x112x1 channel
        .Convolution_done(Convolution_done) // signal to indicate that the processing is done
    );
    
endmodule