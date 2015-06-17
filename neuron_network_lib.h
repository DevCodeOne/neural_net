#define PRINT_DELAY 10000

typedef struct synapse 
{
  float *weight; 
  float *value;
} synapse;

typedef struct output_neuron 
{
  synapse **input;
  int input_count;
  float activity;
} oneuron;

typedef struct hidden_neuron 
{
  synapse **input; 
  synapse **output;
  int input_count;
  int output_count;
  float activity;
} hneuron;

typedef struct neural_network 
{
  oneuron *oneurons; 
  hneuron **hneurons;
  float *output_layer_weights;
  float *output_layer_values; 
  float **hidden_layer_weights; 
  float **hidden_layer_values;
  unsigned int *hidden_layer_size;
  unsigned int hidden_layer_depth;
  unsigned int input_layer_size;
  unsigned int output_layer_size;
} neural_network;

neural_network *build_neural_network(unsigned int input_layer_size, 
  unsigned int *hidden_layer_size, unsigned int hidden_layer_depth, unsigned int output_layer_size);
float *emulate(neural_network *nn, float *input);
float adjust_weights(neural_network *nn, float *input, float *expected_output, float learning_rate);
void teach(neural_network *nn, int number_of_samples, float *inputs, int number_of_inputs, float *expected_outputs, int number_of_outputs, float learning_rate, int passes);
static inline float sigmoid(float x);
static inline float random_weight(float min, float max);