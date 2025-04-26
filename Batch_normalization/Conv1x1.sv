module Conv1x1 #(parameter width = 8 , parameter rows = 3 , parameter cols = 3) (
    input CLK,               
    input RST,             
	input signed [width - 1:0] kernel, // kernel width = 8 with 1x1x1 
    input signed [width - 1:0] pixel_in  [0: rows-1][0: cols-1], // 3x3x1 channel
    output logic signed [width - 1:0] pixel_out [0: rows-1][0: cols-1] // 3x3x1 channel
);


logic signed [2*width - 1:0] conv [0: rows-1][0: cols-1] ;

integer i, j;
always @(posedge CLK or negedge RST ) begin
    if (!RST) begin
		for (i = 0; i < rows ; i = i + 1) begin
			for (j = 0; j < cols ; j = j + 1) begin
				pixel_out[i][j] <= 8'b0;
			end
		end
    end else begin
        for (i = 0; i < rows ; i = i + 1) begin
            for (j = 0; j < cols ; j = j + 1) begin
			
                conv[i][j] <= pixel_in[i][j] * kernel;			
				pixel_out[i][j] <= conv[i][j][width-1 +4 :0 +4 ] ; //middle 8 bits for fixed point 
				
            end           
        end
    end
end

endmodule


