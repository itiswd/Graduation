module Main_Controller (
  input logic clk,
  input logic rst_n,
  input logic [1:0] layer_type,
  input logic start,
  input logic vector_done,
  input logic winograd_done,
  input logic se_done,
  output logic vector_start,
  output logic winograd_start,
  output logic se_start,
  output logic done
);

  typedef enum logic [1:0] {
    IDLE,
    RUN,
    WAIT,
    COMPLETE
  } state_t;

  state_t current_state, next_state;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      current_state <= IDLE;
    else
      current_state <= next_state;
  end

  always_comb begin
    vector_start = 0;
    winograd_start = 0;
    se_start = 0;
    done = 0;

    next_state = current_state;

    case (current_state)
      IDLE: begin
        if (start)
          next_state = RUN;
      end

      RUN: begin
        case (layer_type)
          2'b00: vector_start = 1;
          2'b01: winograd_start = 1;
          2'b10: se_start = 1;
        endcase
        next_state = WAIT;
      end

      WAIT: begin
        case (layer_type)
          2'b00: if (vector_done) next_state = COMPLETE;
          2'b01: if (winograd_done) next_state = COMPLETE;
          2'b10: if (se_done) next_state = COMPLETE;
        endcase
      end

      COMPLETE: begin
        done = 1;
        next_state = IDLE;
      end
    endcase
  end

endmodule
