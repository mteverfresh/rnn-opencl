#include "lstm.hpp"

LSTMCell::LSTMCell(float *forget, float *input, float *internal, float *output)
{
	this.w_forget = forget;
	this.w_input = input;
	this.w_internal = internal;
	this.w_output = output;
}
//think of these like sets of instructions
//for 3 input: S1 S2 D
//for 2 input: S1 D
inline void LSTMCell::forget()
{
	matrixMultiplyCL	(this.concat_input,	this.w_forget, 		this.forget_calc);
	matrixAddCL		(this.forget_calc, 	this.b_forget, 		this.forget_calc);
	sigmoidCL		(this.forget_calc, 	this.forget_calc);
}
inline void LSTMCell::input()
{
	matrixMultiplyCL	(this.concat_input, 	this.w_input, 		this.input_calc);
	matrixAddCL		(this.input_calc, 	this.b_input, 		this.input_calc);
	sigmoidCL		(this.input_calc, 	this.input_calc);
}
inline void LSTMCell::internal()
{
	matrixMultiplyCL	(this.concat_input, 	this.w_internal, 	this.internal_calc);
	matrixAddCL		(this.internal_calc, 	this.b_internal, 	this.internal_calc);
	tanhCL			(this.internal_calc, 	this.internal_calc);
}
inline void LSTMCell::output()
{
	matrixMultiplyCL	(this.concat_input,	this.w_output, 		this.output_calc);
	matrixAddCL		(this.output_calc, 	this.b_output, 		this.output_calc);
	sigmoidCL		(this.output_calc, 	this.output_calc);
}
inline void LSTMCell::nextState()
{
	matrixMultiplyCL	(this.forget_calc, 	this.prev_state, 	this.forget_calc);
	matrixMultiplyCL	(this.input_calc, 	this.internal_calc, 	this.input_calc);
	matrixAddCL		(this.forget_calc,	this.input_calc);
}

