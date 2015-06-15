#include "utils.h"

int sort_backwards_abs_values(double *array_to_sort, int *index_array, int size, double tolerated_error)
{
  int under_tolerated_error = size;
  for (int i = 0; i < size; i++)
  {
    for (int j = i+1; j < size; j++)
    {
      if (fabs(array_to_sort[i]) < fabs(array_to_sort[j]))
      {
      
        double new = array_to_sort[j];
        int new_index = index_array[j];
        for (int k = j; k > i; k--) 
        {
          array_to_sort[k] = array_to_sort[k-1];
          index_array[k] = index_array[k-1];
        }
        array_to_sort[i] = new;
        index_array[i] = new_index;
      }
      if (fabs(array_to_sort[i]) < tolerated_error && under_tolerated_error > i)
      {
        under_tolerated_error = i;
      }
    }
  }
  return under_tolerated_error;
}