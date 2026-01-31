`timescale 1ns/1ns
module processing_element(input wire clk, input wire rst_n, input wire enable, input wire clear_acc,
    input wire signed[7:0] weight_in, input wire weight_load, output wire signed[7:0] weight_out,
    input wire signed[7:0] act_in, output wire signed[7:0] act_out,
    input wire signed[31:0] psum_in, output wire signed[31:0] psum_out);
    reg signed[7:0] weight_reg, act_reg; reg signed[31:0] acc_reg;
    wire signed[15:0] product; assign product = weight_reg * act_reg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin weight_reg <= 0; act_reg <= 0; acc_reg <= 0; end
        else if (enable) begin
            if (weight_load) weight_reg <= weight_in;
            act_reg <= act_in;
            if (clear_acc) acc_reg <= 0; else acc_reg <= psum_in + product;
        end
    end
    assign weight_out = weight_reg; assign act_out = act_reg; assign psum_out = acc_reg;
endmodule

module systolic_array_4x4(input wire clk, input wire rst_n, input wire enable, input wire clear_acc, input wire weight_load,
    input wire signed[7:0] weight_col0, weight_col1, weight_col2, weight_col3,
    input wire signed[7:0] act_row0, act_row1, act_row2, act_row3,
    output wire signed[31:0] result_col0, result_col1, result_col2, result_col3);
    wire signed[7:0] w[0:3][0:3]; wire signed[7:0] a[0:3][0:3]; wire signed[31:0] p[0:3][0:3];
    processing_element pe00(.clk(clk),.rst_n(rst_n),.enable(enable),.clear_acc(clear_acc),.weight_in(weight_col0),.weight_load(weight_load),.weight_out(w[0][0]),.act_in(act_row0),.act_out(a[0][0]),.psum_in(32'sd0),.psum_out(p[0][0]));
    processing_element pe01(.clk(clk),.rst_n(rst_n),.enable(enable),.clear_acc(clear_acc),.weight_in(weight_col1),.weight_load(weight_load),.weight_out(w[0][1]),.act_in(a[0][0]),.act_out(a[0][1]),.psum_in(32'sd0),.psum_out(p[0][1]));
    processing_element pe02(.clk(clk),.rst_n(rst_n),.enable(enable),.clear_acc(clear_acc),.weight_in(weight_col2),.weight_load(weight_load),.weight_out(w[0][2]),.act_in(a[0][1]),.act_out(a[0][2]),.psum_in(32'sd0),.psum_out(p[0][2]));
    processing_element pe03(.clk(clk),.rst_n(rst_n),.enable(enable),.clear_acc(clear_acc),.weight_in(weight_col3),.weight_load(weight_load),.weight_out(w[0][3]),.act_in(a[0][2]),.act_out(a[0][3]),.psum_in(32'sd0),.psum_out(p[0][3]));
    processing_element pe10(.clk(clk),.rst_n(rst_n),.enable(enable),.clear_acc(clear_acc),.weight_in(w[0][0]),.weight_load(weight_load),.weight_out(w[1][0]),.act_in(act_row1),.act_out(a[1][0]),.psum_in(p[0][0]),.psum_out(p[1][0]));
    processing_element pe11(.clk(clk),.rst_n(rst_n),.enable(enable),.clear_acc(clear_acc),.weight_in(w[0][1]),.weight_load(weight_load),.weight_out(w[1][1]),.act_in(a[1][0]),.act_out(a[1][1]),.psum_in(p[0][1]),.psum_out(p[1][1]));
    processing_element pe12(.clk(clk),.rst_n(rst_n),.enable(enable),.clear_acc(clear_acc),.weight_in(w[0][2]),.weight_load(weight_load),.weight_out(w[1][2]),.act_in(a[1][1]),.act_out(a[1][2]),.psum_in(p[0][2]),.psum_out(p[1][2]));
    processing_element pe13(.clk(clk),.rst_n(rst_n),.enable(enable),.clear_acc(clear_acc),.weight_in(w[0][3]),.weight_load(weight_load),.weight_out(w[1][3]),.act_in(a[1][2]),.act_out(a[1][3]),.psum_in(p[0][3]),.psum_out(p[1][3]));
    processing_element pe20(.clk(clk),.rst_n(rst_n),.enable(enable),.clear_acc(clear_acc),.weight_in(w[1][0]),.weight_load(weight_load),.weight_out(w[2][0]),.act_in(act_row2),.act_out(a[2][0]),.psum_in(p[1][0]),.psum_out(p[2][0]));
    processing_element pe21(.clk(clk),.rst_n(rst_n),.enable(enable),.clear_acc(clear_acc),.weight_in(w[1][1]),.weight_load(weight_load),.weight_out(w[2][1]),.act_in(a[2][0]),.act_out(a[2][1]),.psum_in(p[1][1]),.psum_out(p[2][1]));
    processing_element pe22(.clk(clk),.rst_n(rst_n),.enable(enable),.clear_acc(clear_acc),.weight_in(w[1][2]),.weight_load(weight_load),.weight_out(w[2][2]),.act_in(a[2][1]),.act_out(a[2][2]),.psum_in(p[1][2]),.psum_out(p[2][2]));
    processing_element pe23(.clk(clk),.rst_n(rst_n),.enable(enable),.clear_acc(clear_acc),.weight_in(w[1][3]),.weight_load(weight_load),.weight_out(w[2][3]),.act_in(a[2][2]),.act_out(a[2][3]),.psum_in(p[1][3]),.psum_out(p[2][3]));
    processing_element pe30(.clk(clk),.rst_n(rst_n),.enable(enable),.clear_acc(clear_acc),.weight_in(w[2][0]),.weight_load(weight_load),.weight_out(w[3][0]),.act_in(act_row3),.act_out(a[3][0]),.psum_in(p[2][0]),.psum_out(p[3][0]));
    processing_element pe31(.clk(clk),.rst_n(rst_n),.enable(enable),.clear_acc(clear_acc),.weight_in(w[2][1]),.weight_load(weight_load),.weight_out(w[3][1]),.act_in(a[3][0]),.act_out(a[3][1]),.psum_in(p[2][1]),.psum_out(p[3][1]));
    processing_element pe32(.clk(clk),.rst_n(rst_n),.enable(enable),.clear_acc(clear_acc),.weight_in(w[2][2]),.weight_load(weight_load),.weight_out(w[3][2]),.act_in(a[3][1]),.act_out(a[3][2]),.psum_in(p[2][2]),.psum_out(p[3][2]));
    processing_element pe33(.clk(clk),.rst_n(rst_n),.enable(enable),.clear_acc(clear_acc),.weight_in(w[2][3]),.weight_load(weight_load),.weight_out(w[3][3]),.act_in(a[3][2]),.act_out(a[3][3]),.psum_in(p[2][3]),.psum_out(p[3][3]));
    assign result_col0 = p[3][0]; assign result_col1 = p[3][1]; assign result_col2 = p[3][2]; assign result_col3 = p[3][3];
endmodule
