#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#include "sample.h"

double gettimeofday_sec()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char *argv[])
{
  double t1,t2;
  int n = 100;
  if (argc > 1) {
    n = atoi( argv[1] );
  }
  printf("n: %d\n", n);
  t1 = gettimeofday_sec();
  cuda_kernel_exec(n);
  t2 = gettimeofday_sec();
  printf("elapsed: %f sec.\n", t2-t1);
  return 0;
}
