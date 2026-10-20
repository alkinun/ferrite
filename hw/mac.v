// The multiply-accumulate unit (ROADMAP, Stage 1): acc + a * b, a and b bf16, acc fp32, bit for bit as hw/mac.py has
// it. fma is the arithmetic, all combinational; mac is an accumulator around it, loaded or stepped each clock.

module fma (
  input  wire [31:0] c,
  input  wire [15:0] a, b,
  output reg  [31:0] y
);
  wire       sc = c[31], sp = a[15] ^ b[15];
  wire [7:0] ea = a[14:7], eb = b[14:7], ec = c[30:23];
  wire       a_zero = ea == 0, b_zero = eb == 0, c_zero = ec == 0;  // zeros, and denormals flushed to them
  wire       a_inf = &ea, b_inf = &eb, c_inf = &ec;
  wire       nan = a_inf && a[6:0] != 0 || b_inf && b[6:0] != 0 || c_inf && c[22:0] != 0;

  // the product, exact: 16 bits of significand and its biased exponent, which out of fp32's range makes it infinity or zero
  wire [15:0]        pm = {1'b1, a[6:0]} * {1'b1, b[6:0]};
  wire signed [10:0] pe = {3'b0, ea} + {3'b0, eb} + {10'b0, pm[15]} - 11'd127;
  wire [23:0]        psig = pm[15] ? {pm, 8'b0} : {pm[14:0], 9'b0};
  wire               p_inf = a_inf || b_inf || !a_zero && !b_zero && pe >= 255;
  wire               p_zero = a_zero || b_zero || pe <= 0;

  // the sum of two normals: the smaller aligned to the larger with guard, round and sticky bits, added or subtracted,
  // normalized, rounded to nearest even
  wire        c_big = {ec, c[22:0]} >= {pe[7:0], psig[22:0]};
  wire        sx = c_big ? sc : sp, sub = sc ^ sp;
  wire [7:0]  ex = c_big ? ec : pe[7:0], d = c_big ? ec - pe[7:0] : pe[7:0] - ec;
  wire [26:0] mx = {c_big ? {1'b1, c[22:0]} : psig, 3'b0}, my = {c_big ? psig : {1'b1, c[22:0]}, 3'b0};
  wire [26:0] ys = my >> d | {26'b0, |(my & ~(27'h7FFFFFF << d))};  // the bits shifted out, as one sticky bit
  wire [27:0] s = sub ? {1'b0, mx} - {1'b0, ys} : {1'b0, mx} + {1'b0, ys};
  reg  [4:0]  lz;  // leading zeros below a carry
  integer i;
  always @* begin
    lz = 0;
    for (i = 0; i < 27; i = i + 1) if (s[i]) lz = 5'd26 - i[4:0];
  end
  wire [26:0]        n = s[27] ? {s[27:2], s[1] | s[0]} : s[26:0] << lz;
  wire signed [10:0] en = s[27] ? {3'b0, ex} + 11'd1 : {3'b0, ex} - {6'b0, lz};
  wire [24:0]        r = {1'b0, n[26:3]} + {24'b0, n[2] && (n[1] || n[0] || n[3])};
  wire signed [10:0] er = en + {10'b0, r[24]};
  wire [31:0]        sum = s == 0 ? 32'h0 : er >= 255 ? {sx, 8'hFF, 23'b0} : er <= 0 ? {sx, 31'b0} : {sx, er[7:0], r[24] ? r[23:1] : r[22:0]};

  always @* begin
    if (nan || (a_inf || b_inf) && (a_zero || b_zero)) y = 32'h7FC00000;  // every NaN out is this one
    else if (p_inf) y = c_inf && sc != sp ? 32'h7FC00000 : {sp, 8'hFF, 23'b0};
    else if (c_inf) y = c;
    else if (p_zero) y = c_zero ? {sc & sp, 31'b0} : c;
    else if (c_zero) y = {sp, pe[7:0], psig[22:0]};
    else y = sum;
  end
endmodule

module mac (
  input  wire        clk,
  input  wire        load,  // acc becomes init
  input  wire        en,    // else acc becomes acc + a * b
  input  wire [31:0] init,
  input  wire [15:0] a, b,
  output reg  [31:0] acc
);
  wire [31:0] next;
  fma step (.c(acc), .a(a), .b(b), .y(next));
  always @(posedge clk) if (load) acc <= init; else if (en) acc <= next;
endmodule
