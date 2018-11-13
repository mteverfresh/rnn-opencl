#ifndef LSTM_H
#define LSTM_H

#include "oclabstract.hpp"

class LSTMCell
{
	private:
		//weight memory
		cl_float *w_forget;
		cl_float *w_input;
		cl_float *w_internal;
		cl_float *w_output;
		//bias memory
		cl_float *b_forget;
		cl_float *b_input;
		cl_float *b_internal;
		cl_float *b_output;

		//intermediate matrices
		cl_float *forget_calc;
		cl_float *input_calc;
		cl_float *internal_calc;
		cl_float *concat_input;

		//cell IO
		cl_float *curr_input;
		cl_float *curr_output;
		cl_float *prev_output;
		cl_float *curr_state;
		cl_float *prev_state;

		//private gate functions
		inline void forget();
		inline void input();
		inline void internal();
		inline void output();
		inline void nextState();
		inline void nextOutput();
	public:
		LSTMCell(cl_float*, cl_float*, cl_float*, cl_float*);
		void forwardPass(cl_float *new_input);
		void backwardPass(cl_float *training_input);
}
#endif
