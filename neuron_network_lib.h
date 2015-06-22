#define PRINT_DELAY 10000

typedef struct synapse 
{
  double *weight; 
  double *value;
} synapse;

typedef struct output_neuron 
{
  synapse **input;
  int input_count;
  double activity;
} oneuron;

typedef struct hidden_neuron 
{
  synapse **input; 
  synapse **output;
  int input_count;
  int output_count;
  double activity;
} hneuron;

typedef struct neural_network 
{
  oneuron *oneurons; 
  hneuron **hneurons;
  double *output_layer_weights;
  double *output_layer_values; 
  double **hidden_layer_weights; 
  double **hidden_layer_values;
  unsigned int *hidden_layer_size;
  unsigned int hidden_layer_depth;
  unsigned int input_layer_size;
  unsigned int output_layer_size;
} neural_network;

neural_network *build_neural_network(unsigned int input_layer_size, 
  unsigned int *hidden_layer_size, unsigned int hidden_layer_depth, unsigned int output_layer_size);
double *emulate(neural_network *nn, double *input);
double adjust_weights(neural_network *nn, double *input, double *expected_output, double learning_rate);
void teach(neural_network *nn, int number_of_samples, double *inputs, int number_of_inputs, double *expected_outputs, int number_of_outputs, double learning_rate, int passes);
static inline double sigmoid(double x);
static inline double random_weight(double min, double max);