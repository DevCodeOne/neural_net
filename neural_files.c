#include <stdio.h>
#include <stdlib.h>
#include "neuron_network_lib.h"
#include "neural_files.h"

unsigned int read_int(FILE *file)
{
  return (unsigned int) (fgetc(file) << 0 | fgetc(file) << 8 | fgetc(file) << 16 | fgetc(file) << 24);
}

// stupid fukks
unsigned int read_int_big_endian(FILE *file) 
{ 
  return (unsigned int) (fgetc(file) << 24 | fgetc(file) << 16 | fgetc(file) << 8 | fgetc(file) << 0);
}

void write_int(unsigned int i, FILE *file) 
{
  fwrite(&i, sizeof(int), 1, file);
}

test_sample *read_test_samples(char *name_of_samples, char *name_of_answers)
{
  FILE *samples = fopen(name_of_samples, "rb");
  FILE *answers = fopen(name_of_answers, "rb");
  test_sample *ts = malloc(sizeof(test_sample));
  read_int_big_endian(samples); // skip magic numbr
  read_int_big_endian(answers); read_int_big_endian(answers);
  ts->number_of_samples = read_int_big_endian(samples);
  ts->width_of_samples = read_int_big_endian(samples); 
  ts->height_of_samples = read_int_big_endian(samples);
  ts->data = malloc(sizeof(char) * ts->number_of_samples * ts->width_of_samples * ts->height_of_samples);
  ts->answers = malloc(sizeof(char) * ts->number_of_samples);
  
  fread(ts->data, sizeof(char), ts->number_of_samples * ts->width_of_samples * ts->height_of_samples, samples);
  fread(ts->answers, sizeof(char), ts->number_of_samples, answers);
  fclose(samples);
  fclose(answers);
  return ts;
}

void write_neural_network_to_file(neural_network *nn, char *str)
{
  FILE *file = fopen(str, "wb");
  if (file == NULL)
  {
    printf("Error opening file %s \n", str);
    return;
  }
  write_int(nn->input_layer_size, file); 
  write_int(nn->hidden_layer_depth, file); 
  for (int i = 0; i < nn->hidden_layer_depth; i++)
    write_int(nn->hidden_layer_size[i], file); 
    
  write_int(nn->output_layer_size, file);
  
  for (int i = 0; i < nn->hidden_layer_depth; i++)
  {
    int len = i > 0 ? nn->hidden_layer_size[i-1] : nn->input_layer_size;
    fwrite(nn->hidden_layer_weights[i], sizeof(double), nn->hidden_layer_size[i] * len, file);
  }
  
  fwrite(nn->output_layer_weights, sizeof(double), nn->hidden_layer_size[nn->hidden_layer_depth-1] * nn->output_layer_size, file);
  fclose(file);
}

neural_network *read_neural_network_from_file(char *str)
{
  FILE *file = fopen(str, "rb");
  if (file == NULL)
  {
    printf("Error opening file %s \n", str);
    return NULL;
  }
  
  unsigned int input_layer_size = read_int(file);
  unsigned int hidden_layer_depth = read_int(file);
  unsigned int *hidden_layer_size = malloc(sizeof(unsigned int) * hidden_layer_depth); 
  
  for (int i = 0; i < hidden_layer_depth; i++)
    hidden_layer_size[i] = read_int(file); 
  
  unsigned int output_layer_size = read_int(file);
  
  neural_network *nn = build_neural_network(input_layer_size, hidden_layer_size, hidden_layer_depth, output_layer_size);
  
  printf("input_layer_size : %d \nhidden_layer_depth %d \n", input_layer_size, hidden_layer_depth);
  for (int i = 0; i < hidden_layer_depth; i++)
    printf(" hidden_layer_size[%d] : %d \n", i, hidden_layer_size[i]);
  
  printf("output_layer_size : %d \n", output_layer_size);
  
  for (int i = 0; i < nn->hidden_layer_depth; i++)
  {
    int len = i > 0 ? nn->hidden_layer_size[i-1] : nn->input_layer_size;
    fread(nn->hidden_layer_weights[i], sizeof(double), nn->hidden_layer_size[i] * len, file);
  }
  
  fread(nn->output_layer_weights, sizeof(double), nn->hidden_layer_size[nn->hidden_layer_depth-1] * nn->output_layer_size, file);
  fclose(file);
  return nn;
}