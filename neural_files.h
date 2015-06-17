typedef struct test_sample 
{
  int number_of_samples;
  int width_of_samples; 
  int height_of_samples; 
  unsigned char *data;
  unsigned char *answers;
} test_sample;

unsigned int read_int(FILE *file);
unsigned int read_int_big_endian(FILE *file);
void write_int(unsigned int i, FILE *file);
test_sample *read_test_samples(char *name_of_samples, char *name_of_answers);
void write_neural_network_to_file(neural_network *nn, char *str);
neural_network *read_neural_network_from_file(char *str);