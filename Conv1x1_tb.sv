`timescale 1ns/1ps

module Conv1x1_tb();
	localparam SF = 2.0**-4.0;  // Q4.4 scaling factor is 2^-4
	
	parameter width = 8 ;
	parameter rows  = 3 ;
	parameter cols  = 3 ;

	logic CLK;               
	logic RST;       
	logic signed [width - 1:0]kernel;	
	logic signed [width - 1:0] pixel_in  [0: rows-1][0: cols-1]; // 3x3x1 channel
	logic signed [width - 1:0] pixel_out [0: rows-1][0: cols-1]; // 3x3x1 channel



	Conv1x1 #(.width(width),.rows(rows),.cols(cols)) DUT (
		.CLK(CLK),
		.RST(RST),
		.kernel(kernel),
		.pixel_in(pixel_in),
		.pixel_out(pixel_out)
	);

always #5 CLK = ~CLK ;

integer i,j ;
    initial begin
      
	   $dumpfile("Conv1x1.vcd");
       $dumpvars;
	   
	    CLK = 0 ;
	    RST = 0 ;
		kernel = 8'b0;
		
		#10 RST = 1;
		kernel = 8'b0000_1000; 
		// Initialize input image (example values)
		pixel_in[0][0] = 8'b0001_0000; // 1.0
		pixel_in[0][1] = 8'b0001_1000; // 1.5
		pixel_in[0][2] = 8'b0000_1000; // 0.5
		pixel_in[1][0] = 8'b1111_0000; // -1.0
		pixel_in[1][1] = 8'b0000_0000; // 0.0
		pixel_in[1][2] = 8'b0001_1100; // 1.75
		pixel_in[2][0] = 8'b0000_0100; // 0.25
		pixel_in[2][1] = 8'b1110_1000; // -1.5
		pixel_in[2][2] = 8'b0001_1110; // 1.875
		
		
		#20
		
		$display("Test Case 1 (Kernel=0.5)");
    for (int i = 0; i < rows; i++) begin
        for (int j = 0; j < cols; j++) begin
            $write("%f ", $itor(pixel_out[i][j] * SF));
        end
		$display("");
    end
		
		#20 
		kernel = 8'b0001_0000; 
		
		#20
		$display("Test Case 2 (Kernel=1.0)");
    for (int i = 0; i < rows; i++) begin
        for (int j = 0; j < cols; j++) begin
            $write("%f ", $itor(pixel_out[i][j] * SF));
        end
		$display("");
    end
	
	    #100
		$stop;
	   
    end

endmodule