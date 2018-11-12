#ifndef LSTM_H
#define LSTM_H

class LSTMCell
{
	private:
		//weight memory
		float *w_forget;
		float *w_input;
		float *w_internal;
		float *w_output;

		//intermediate matrices
		float *forget_calc;
		float *input_calc;
		float *internal_calc;
		float *concat_input;

		//cell IO
		float *curr_input;
		float *curr_output;
		float *prev_output;
		float *curr_state;
		float *prev_state;

		//private gate functions
		inline void forget();
		inline void input();
		inline void internal();
		inline void output();
		inline void nextState();
		inline void nextOutput();
	public:
		LSTMCell();
		void forwardPass(float *new_input);
		void backwardPass(float *training_input);
}

//Other function prototypes
bool setupOclEnv(char *kernel_file);
void matrixMultiplyCl(float *a, float *b, float *output);
void matrixAddCl(float *a, float *b, float *output);
void sigmoidCl(float *in, float *out);
void tanhCl(float *in, float *out);
void matrixConcatCl(float *a, float *b, float *output);

#endif
