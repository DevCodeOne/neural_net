#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include "neuron_network_lib.h"

static inline double sigmoid(double x) { return  0.5 * x / (1 + (x > 0 ? x : -x)) + 0.5; }
static inline double random_weight(double min, double max) { return (double) ((rand() / RAND_MAX) * (max - min)) + min; }

neural_network *build_neural_network(unsigned int input_layer_size, unsigned int *hidden_layer_size, 
unsigned int hidden_layer_depth, unsigned int output_layer_size)
{
  srand(time(NULL));
  int last_layer_index = hidden_layer_depth-1;
  neural_network *nn = malloc(sizeof(neural_network));
  nn->oneurons = malloc(sizeof(oneuron) * output_layer_size);
  nn->hneurons = malloc(sizeof(hneuron *) * hidden_layer_depth);
  nn->hidden_layer_size = malloc(sizeof(double) * hidden_layer_depth); 
  for (int i = 0; i < hidden_layer_depth; i++)
    nn->hidden_layer_size[i] = hidden_layer_size[i];
  
  nn->hidden_layer_depth = hidden_layer_depth; 
  nn->input_layer_size = input_layer_size; 
  nn->output_layer_size = output_layer_size;
  
  nn->output_layer_weights = malloc(sizeof(double) * output_layer_size * hidden_layer_size[last_layer_index]); 
  nn->output_layer_values = malloc(sizeof(double) * output_layer_size * hidden_layer_size[last_layer_index]);
  
  nn->hidden_layer_weights = malloc(sizeof(double *) * hidden_layer_depth);
  nn->hidden_layer_values = malloc(sizeof(double *) * hidden_layer_depth);
  for (int i = 0; i < hidden_layer_depth; i++)
  {
    nn->hidden_layer_weights[i] = malloc(sizeof(double) * hidden_layer_size[i] * (i-1 > 0 ? hidden_layer_size[i-1] : input_layer_size));
    nn->hidden_layer_values[i] = malloc(sizeof(double) * hidden_layer_size[i] * (i-1 > 0 ? hidden_layer_size[i-1] : input_layer_size));
  }
  
  for (int i = 0; i < hidden_layer_depth; i++)
  {
    nn->hneurons[i] = malloc(sizeof(hneuron) * hidden_layer_size[i]);
    for (int j = 0; j < hidden_layer_size[i]; j++)
    {
      nn->hneurons[i][j].input = malloc(sizeof(synapse *) * (i > 1 ? hidden_layer_size[i-1] : input_layer_size));
      nn->hneurons[i][j].output = malloc(sizeof(synapse *) * ((i+1) < hidden_layer_depth ? hidden_layer_size[i+1] : output_layer_size));
    }
  }
  
  for (int i = 0; i < output_layer_size; i++) 
  {
    nn->oneurons[i].input = malloc(sizeof(synapse *) * hidden_layer_size[last_layer_index]);
    nn->oneurons[i].input_count = 0;
  }
  
  for (int i = 0; i < input_layer_size; i++) 
  {
    synapse **syn = malloc(sizeof(synapse *) * hidden_layer_size[0]);
    for (int j = 0; j < hidden_layer_size[0]; j++)
    {
      syn[j] = malloc(sizeof(synapse));
      nn->hneurons[0][j].input[nn->hneurons[0][j].input_count++] = syn[j];
      syn[j]->weight = &nn->hidden_layer_weights[0][(j*input_layer_size)+i];
      syn[j]->value = &nn->hidden_layer_values[0][(j*input_layer_size)+i];
      nn->hidden_layer_weights[0][(j*input_layer_size)+i] = random_weight(-0.5, 0.5);
      nn->hidden_layer_values[0][(j*input_layer_size)+i] = 0;
    }
  }
  
  for (int i = 0; i < last_layer_index; i++)
  {
    for (int j = 0; j < hidden_layer_size[i]; j++)
    {
      synapse **syn = malloc(sizeof(synapse *) * hidden_layer_size[0]);
      for (int k = 0; k < hidden_layer_size[i+1]; k++)
      {
        syn[k] = malloc(sizeof(synapse));
        nn->hneurons[i+1][k].input[nn->hneurons[i+1][k].input_count++] = syn[k];
        nn->hneurons[i][j].output[nn->hneurons[i][j].output_count] = syn[k];
        nn->hidden_layer_weights[i][(k*hidden_layer_size[i-1])+j] = random_weight(-0.5, 0.5);
        nn->hidden_layer_values[i][(k*hidden_layer_size[i-1])+j] = 0;
        nn->hneurons[i][j].output[nn->hneurons[i][j].output_count]->weight = &nn->hidden_layer_weights[i][(k*hidden_layer_size[i-1])+j];
        nn->hneurons[i][j].output[nn->hneurons[i][j].output_count++]->value = &nn->hidden_layer_values[i][(k*hidden_layer_size[i-1])+j];
      }
    }
  }
  
  for (int i = 0; i < hidden_layer_size[hidden_layer_depth-1]; i++)
  {
    synapse **syn = malloc(sizeof(synapse *) * hidden_layer_size[0]);
    for (int j = 0; j < output_layer_size; j++)
    {
      syn[j] = malloc(sizeof(synapse));
      nn->oneurons[j].input[nn->oneurons[j].input_count++] = syn[j];
      nn->hneurons[last_layer_index][i].output[nn->hneurons[last_layer_index][i].output_count] = syn[j]; 
      nn->output_layer_weights[(j*hidden_layer_size[last_layer_index])+i] = random_weight(-0.5, 0.5);
      nn->output_layer_values[(j*hidden_layer_size[last_layer_index])+i] = 0;
      nn->hneurons[last_layer_index][i].output[nn->hneurons[last_layer_index][i].output_count]->weight = &nn->output_layer_weights[(j*hidden_layer_size[last_layer_index])+i];
      nn->hneurons[last_layer_index][i].output[nn->hneurons[last_layer_index][i].output_count++]->value = &nn->output_layer_values[(j*hidden_layer_size[last_layer_index])+i];
    }
  }
  return nn;
}

double *emulate(neural_network *nn, double *input)
{
  const unsigned int stride = sizeof(double) * nn->input_layer_size;
  register unsigned int i, j, k;
  register double value;
  register double activity; 
  hneuron *neuron;
  unsigned int len;
  unsigned int off;
  double *base_pointer;
  
  double *output = malloc(sizeof(double) * nn->output_layer_size);
  
  len = nn->hidden_layer_size[0];
  base_pointer = nn->hidden_layer_values[0];
  
  #pragma omp parallel for private(i, j) default(shared) schedule(static)
  for (i = 0; i < len;i++)
    memcpy((base_pointer+i*stride), input, stride);
  
  for (i = 0; i < nn->hidden_layer_depth; i++)
  {
    len = i != 0 ? nn->hidden_layer_size[i-1] : nn->input_layer_size;
    #pragma omp parallel for private(j, k, value, off, activity, neuron) default(shared) schedule(static)
    for (j = 0; j < nn->hidden_layer_size[i]; j++)
    {
      value = 0;
      off = (j*len);
      
      neuron = &nn->hneurons[i][j];
      
      for (k = 0; k < len; k++)
        value += nn->hidden_layer_weights[i][off+k] * nn->hidden_layer_values[i][off+k];
      activity = sigmoid(value);
      neuron->activity = activity; 
      for (k = 0; k < nn->hneurons[i][j].output_count; k++)
        (*neuron->output[k]->value) = activity;
    }
  }
  
  #pragma omp parallel for private(i, j, value, off, activity) default(shared) schedule(static)
  for (i = 0; i < nn->output_layer_size; i++)
  {
    value = 0;
    off = (i*nn->hidden_layer_size[nn->hidden_layer_depth-1]);
    
    for (j = 0; j < nn->oneurons[i].input_count; j++)
      value += (nn->output_layer_values[off+j] * nn->output_layer_weights[off+j]);
    
    activity = sigmoid(value);
    nn->oneurons[i].activity = activity;
    output[i] = nn->oneurons[i].activity;
  }
  return output;
}

double adjust_weights(neural_network *nn, double *input, double *expected_output, double learning_rate)
{
  double *output = emulate(nn, input);
  double **error = malloc(sizeof(double *) * (nn->hidden_layer_depth+1));
  double error_ges = 0;
  oneuron *oneurons = nn->oneurons; 
  hneuron **hneurons = nn->hneurons;
  
  register int i, j, k;
  register int count = 0;
  
  for (i = nn->hidden_layer_depth+1; i--; )
    error[i] = malloc(sizeof(double *) * (i < nn->hidden_layer_depth ? nn->hidden_layer_size[i] : nn->output_layer_size));
  
  #pragma omp parallel for private(i) default(shared) schedule(static)
  for (i = 0; i < nn->output_layer_size; i++) 
  {
    error[nn->hidden_layer_depth][i] = output[i] * (1 - output[i]) * (expected_output[i] - output[i]);
    error_ges += fabs(error[nn->hidden_layer_depth][i]);
  }
  
  #pragma omp parallel for private(i, j) default(shared) schedule(static)
  for (i = 0; i < nn->output_layer_size; i++)
  {
    double err = learning_rate * error[nn->hidden_layer_depth][i];
    for (j = 0; j < nn->hidden_layer_size[nn->hidden_layer_depth-1]; j++)
      (*oneurons[i].input[j]->weight) += (err * hneurons[nn->hidden_layer_depth-1][j].activity);
  }
  
  for (i = nn->hidden_layer_depth-1; i >= 0; i--)
  {
    
    #pragma omp parallel for private(j, k, count) default(shared) schedule(static)
    for (j = 0; j < nn->hidden_layer_size[i]; j++)
    {
      double err_backprop = 0;
      count = i+1 != nn->hidden_layer_depth ? nn->hidden_layer_size[i+1] : nn->output_layer_size;

      for (k = 0; k < count; k++)
        err_backprop += (error[i+1][k] * (*hneurons[i][j].output[k]->weight));
      
      error[i][j] = nn->hneurons[i][j].activity * (1 - hneurons[i][j].activity) * err_backprop;
      count = i != 0 ? nn->hidden_layer_size[i-1] : nn->input_layer_size;

      double err = learning_rate * error[i][j];
      
      for (k = 0; k < count; k++)
        (*hneurons[i][j].input[k]->weight) += (err * (*hneurons[i][j].input[k]->value));
    }
  }
  free(output);
  for (int i = 0; i < nn->hidden_layer_depth+1; i++)
    free(*(error+i));
  free(error);
  return error_ges;
}

void teach(neural_network *nn, int number_of_samples, double *inputs, int number_of_inputs, double *expected_outputs, int number_of_outputs, double learning_rate, int passes)
{
  int print_after = number_of_samples / PRINT_DELAY;
  double percent = 0;
  
  int len = 0;
  for (int i = 0; i < passes; i++)
  {
    len = number_of_samples;
    for (int j = 0; j < len; j++)
    {
      adjust_weights(nn, (inputs+j*number_of_inputs), (expected_outputs+j*number_of_outputs), learning_rate);
      if (print_after-- == 0)
      {
        print_after = number_of_samples / PRINT_DELAY;
        percent = j / (double)len;
        printf("Pass %d of %d [ %.2f % ] \r", i+1, passes, (percent * 100));
      }
    }
    percent = 0;
    printf("Pass %d of %d [ %.2f % ] %d elements checked of %d elements overall \n", i+1, passes, 100.0, len, number_of_samples);
  }
}