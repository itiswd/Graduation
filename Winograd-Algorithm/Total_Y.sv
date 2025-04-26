module Total_Y #(parameter width = 16 , parameter rows = 224 , parameter cols = 224) (
    input clk,               
    input rst_n,  
    input divide_done,           
    input signed [width - 1:0] data_out_even_even [0: rows/2 -1][0: cols/2 -1], // 111x111x1 channel
    input signed [width - 1:0] data_out_even_odd [0: rows/2 -1][0: cols/2 -1], // 111x111x1 channel
    input signed [width - 1:0] data_out_odd_even [0: rows/2 -1][0: cols/2 -1], // 111x111x1 channel
    input signed [width - 1:0] data_out_odd_odd [0: rows/2 -1][0: cols/2 -1], // 111x111x1 channel
    input signed [width - 1:0] kernel_even_even [0:2][0:2],
    input signed [width - 1:0] kernel_even_odd [0:2][0:2],
    input signed [width - 1:0] kernel_odd_even [0:2][0:2],   
    input signed [width - 1:0] kernel_odd_odd [0:2][0:2],
    output logic signed [width - 1:0] data_out [0: rows/2 -1][0: cols/2 -1], // 111x111x1 channel
    output logic Convolution_done // signal to indicate that the processing is done
);
    logic signed [width - 1:0] temp_out_even_even [0: rows/2 -1][0: cols/2 -1];
    logic signed [width - 1:0] temp_out_even_odd [0: rows/2 -1][0: cols/2 -1];
    logic signed [width - 1:0] temp_out_odd_even [0: rows/2 -1][0: cols/2 -1];
    logic signed [width - 1:0] temp_out_odd_odd [0: rows/2 -1][0: cols/2 -1];
    integer i, j;

    generate
    Tail_filter_element_wise_Y #(.width(width)) inst_even_even (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(data_out_even_even),
        .kernel(kernel_even_even),
        .data_out(temp_out_even_even)
    );

    Tail_filter_element_wise_Y #(.width(width)) inst_even_odd (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(data_out_even_odd),
        .kernel(kernel_even_odd),
        .data_out(temp_out_even_odd)
    );

    Tail_filter_element_wise_Y #(.width(width)) inst_odd_even(
        .clk(clk),
        .rst_n(rst_n),
        .data_in(data_out_odd_even),
        .kernel(kernel_odd_even),
        .data_out(data_out_odd_even)
    );

    Tail_filter_element_wise_Y #(.width(width)) inst_odd_odd(
        .clk(clk),
        .rst_n(rst_n),
        .data_in(data_out_odd_odd),
        .kernel(kernel_odd_odd),
        .data_out(temp_out_odd_odd)
    );
    always @(posedge clk or negedge clk) begin
        if (!rst_n) begin
            data_out <= '{default : 0} ;
            Convolution_done <= 0;
        end else if (divide_done) begin
            for (i = 0 ; i<rows/2 ; i=i+1 ) begin
                for (j = 0 ; j<cols/2 ; j=j+1 ) begin
                    data_out[i][j] <= temp_out_even_even[i][j] + temp_out_even_odd[i][j] + temp_out_odd_even[i][j] + temp_out_odd_odd[i][j];
                end
            end
            Convolution_done <= 1;
        end else begin
            Convolution_done <= 0;
        end
    end
    endgenerate
endmodule