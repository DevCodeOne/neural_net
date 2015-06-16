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
  return ts;
}

void write_neural_network_to_file(neural_network *nn, char *str)
{
  FILE *file = fopen(str, "wb");
  if (file == NULL)
    printf("Error opening file %s \n", str);
  write_int(nn->input_layer_size, file); 
  write_int(nn->hidden_layer_depth, file); 
  for (int i = 0; i < nn->hidden_layer_depth; i++)
    write_int(nn->hidden_layer_size[i], file); 
  write_int(nn->output_layer_size, file);
  for (int i = 0; i < nn->input_layer_size; i++)
  {
    for (int j = 0; j < nn->ineurons[i].output_count; j++)
    {
      write_synapse(nn->ineurons[i].output[j], file);
    }
  }
  
  for (int i = 0; i < nn->hidden_layer_depth; i++)
  {
    for (int j = 0; j < nn->hidden_layer_size[i]; j++)
    {
      for (int k = 0; k < nn->hneurons[i][j].output_count; k++)
      {
        write_synapse(nn->hneurons[i][j].output[k], file);
      }
    }
  }
  fclose(file);
}

neural_network *read_neural_network_from_file(char *str)
{
  FILE *file = fopen(str, "rb");
  if (file == NULL)
    printf("Error opening file %s \n", str);
  
  unsigned int input_layer_size = read_int(file);
  unsigned int hidden_layer_depth = read_int(file);
  unsigned int *hidden_layer_size = malloc(sizeof(int) * hidden_layer_depth); 
  
  for (int i = 0; i < hidden_layer_depth; i++)
    hidden_layer_size[i] = read_int(file); 
  
  unsigned int output_layer_size = read_int(file);
  
  neural_network *nn = build_neural_network(input_layer_size, hidden_layer_size, hidden_layer_depth, output_layer_size);
  
  printf("input_layer_size : %d \nhidden_layer_depth %d \n", input_layer_size, hidden_layer_depth);
  for (int i = 0; i < hidden_layer_depth; i++)
    printf(" hidden_layer_size[%d] : %d \n", i, hidden_layer_size[i]);
  
  printf("output_layer_size : %d \n", output_layer_size);
  for (int i = 0; i < input_layer_size; i++)
  {
    for (int j = 0; j < hidden_layer_size[0]; j++)
    {
      read_synapse(nn->ineurons[i].output[j], file);
    }
  }
  
  for (int i = 0; i < hidden_layer_depth-1; i++)
  {
    for (int j = 0; j < hidden_layer_size[i]; j++)
    {
      for (int k = 0; k < hidden_layer_size[i+1]; k++)
      {
        read_synapse(nn->hneurons[i][j].output[k], file);
      }
    }
  }
  
  for (int i = 0; i < hidden_layer_size[hidden_layer_depth-1]; i++)
  {
    for (int j = 0; j < output_layer_size; j++)
    {
      read_synapse(nn->hneurons[hidden_layer_depth-1][i].output[j], file);
    }
  }
  
  fclose(file);
  free(hidden_layer_size);
  return nn;
}

static __inline void write_synapse(synapse *syn, FILE *file)
{
  fwrite((void *)(syn->weight), sizeof(double), 1, file);
  fwrite((void *)(syn->value), sizeof(double), 1, file);
}

static __inline void read_synapse(synapse *syn, FILE *file)
{
  fread((void *) (syn->weight), sizeof(double), 1, file); 
  fread((void *) (syn->value), sizeof(double), 1, file);
}