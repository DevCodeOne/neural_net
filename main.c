#include <string.h>
#include <stdio.h>
#include <time.h>
#include "neuron_network_lib.h"
#include "neural_files.h"
#include "utils.h"

#define REPEAT 1
#define ERROR_TOLERANCE 0.4
#define DIGITS_TO_TRAIN 50000
#define DIGITS_TO_TEST 1000

int main(int argc, char *argv[])
{
  time_t start; 
  time_t end;
  
  time(&start);
  
  int hidden_size[1] = {150};
  
  test_sample *ts = read_test_samples("train-images", "train-labels");
  
  double *output = malloc(sizeof(double) * ts->number_of_samples * 10);
  double *input = malloc(sizeof(double) * ts->number_of_samples * ts->width_of_samples * ts->height_of_samples);
  
  printf("Number of items %d width[%d] height[%d] \n", ts->number_of_samples, ts->width_of_samples, ts->height_of_samples);
  
  //neural_network *nn = build_neural_network(ts->width_of_samples * ts->height_of_samples, hidden_size, 1, 10);
  
  neural_network *nn = read_neural_network_from_file("nn.hex");
  
  int len = ts->number_of_samples * ts->width_of_samples * ts->height_of_samples;
  for (int i = 0; i < len; i++)
  {
    if (ts->data[i] < 50)
      input[i] = 0; 
    else 
      input[i] = 1;
  }
  
  for (int i = 0; i < ts->number_of_samples; i++)
  {
    output[0] = 0;
  }
  
  for (int i = 0; i < ts->number_of_samples; i++)
  {
    int off = (i*10); 
    for (int j = 0; j < 10; j++)
    {
      if (j == ts->answers[i])
        output[j+off] = 1;
    }
  }
  
  
  printf("Repeat learning procedure %d times \n", REPEAT);
  
  teach(nn, DIGITS_TO_TRAIN, input, ts->width_of_samples * ts->height_of_samples, output, 10, 0.5, REPEAT);
  
  time(&end);
  
  printf("It took %f seconds to train the neural network \n", difftime(end, start));
  
  double *out = emulate(nn, input);
  
  int error_count = 0;
  printf("Testing %d digits \n", DIGITS_TO_TEST);
  for (int i = DIGITS_TO_TRAIN; i < DIGITS_TO_TEST+DIGITS_TO_TRAIN; i++)
  {
    double *out = emulate(nn, input+(i*ts->width_of_samples * ts->height_of_samples));
    for (int j = 0; j < 10; j++)
    {
      if (fabs(out[j] - output[(i*10)+j]) > ERROR_TOLERANCE)
      {
        error_count++;
        printf("error %f \n", fabs(out[j] - output[(i*10)+j]));
        for (int k = 0; k < 10; k++)
        {
          if (fabs(out[k] - output[(i*10)+k]) > ERROR_TOLERANCE)
            printf("--> output : %f, expected output : %f \n", out[k], output[(i*10)+k]);
          else 
            printf("    output : %f, expected output : %f \n", out[k], output[(i*10)+k]);
        }
        break;
      }
    }
  }
  printf("%d errors found with %d digits checked \n", error_count, DIGITS_TO_TEST);
  
  //write_neural_network_to_file(nn, "nn.hex");
  
  return 0;
}