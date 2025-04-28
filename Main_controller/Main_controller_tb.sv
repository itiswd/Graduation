module tb_Main_Controller;

  // DUT I/O
  logic clk, rst_n;
  logic [1:0] layer_type;
  logic start;
  logic vector_done, winograd_done, se_done;
  logic vector_start, winograd_start, se_start;
  logic done;

  // Instantiate DUT
  Main_Controller uut (
    .clk(clk),
    .rst_n(rst_n),
    .layer_type(layer_type),
    .start(start),
    .vector_done(vector_done),
    .winograd_done(winograd_done),
    .se_done(se_done),
    .vector_start(vector_start),
    .winograd_start(winograd_start),
    .se_start(se_start),
    .done(done)
  );

  // Clock
  always #5 clk = ~clk;

  // Reset task
  task reset_dut();
    begin
      rst_n = 0;
      start = 0;
      layer_type = 2'b00;
      vector_done = 0;
      winograd_done = 0;
      se_done = 0;
      #20;
      rst_n = 1;
      #10;
    end
  endtask

  // Start layer
  task start_layer(input [1:0] ltype);
    begin
      layer_type = ltype;
      start = 1;
      #10;
      start = 0;
      #10; // FSM enters RUN
    end
  endtask

  // Pulse checker: waits for 1-cycle assertion
  task wait_for_start_pulse(input string signal_name, input logic signal);
    begin
      if (signal !== 1) begin
        $display("[ERROR] %s not asserted as 1-cycle pulse at time %0t", signal_name, $time);
        $fatal;
      end else begin
        $display("[PASS] %s pulse detected at time %0t", signal_name, $time);
      end
    end
  endtask

  // Done checker
  task check_done();
    begin
      if (done !== 1) begin
        $display("[ERROR] Done not asserted at time %0t", $time);
        $fatal;
      end else begin
        $display("[PASS] Done asserted at time %0t", $time);
      end
    end
  endtask

  // Main simulation
  initial begin
    $display("\n=== Starting Main Controller Testbench ===\n");
    clk = 0;
    reset_dut();

    // === Test 1: Vector Processor ===
  // === Test 1: Vector Processor ===
$display("[TEST] Vector Processor Path (1x1 Conv)");
start_layer(2'b00);
#10; // allow FSM to move to RUN
#1;  // small delay into that cycle
wait_for_start_pulse("vector_start", vector_start);


    // === Test 2: Winograd Conv ===
    $display("\n[TEST] Winograd Convolution Path (3x3/5x5)");
    start_layer(2'b01);
    wait_for_start_pulse("winograd_start", winograd_start);
    #10;
    winograd_done = 1;
    #10;
    check_done();
    winograd_done = 0;

    // === Test 3: Squeeze-and-Excite ===
    $display("\n[TEST] Squeeze-and-Excite Path");
    start_layer(2'b10);
    wait_for_start_pulse("se_start", se_start);
    #10;
    se_done = 1;
    #10;
    check_done();
    se_done = 0;

    $display("\n??? ALL TEST CASES PASSED ???\n");
    #20;
    $finish;
  end

endmodule
