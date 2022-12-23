#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <openblas/lapacke.h>
//#include <mkl_lapacke.h>
#include <omp.h>

double *generate_matrix(int size)
{
  int i;
  double *matrix = (double *) malloc(sizeof(double) * size * size);

  srand(1);

  for (i = 0; i < size * size; i++) {
    matrix[i] = rand() % 100;
  }

  return matrix;
}

int is_nearly_equal(double x, double y)
{
  const double epsilon = 1e-5 /* some small number */;
  return abs(x - y) <= epsilon * abs(x);
  // see Knuth section 4.2.2 pages 217-218
}

int check_result(double *bref, double *b, int size)
{
  int i;

  for(i = 0; i < size*size; i++) {
    if (!is_nearly_equal(bref[i], b[i]))
      return 0;
  }

  return 1;
}

int my_dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb)
{
 
        double *identidad = (double *) malloc(sizeof(double) * n * n);
        double *matrix_x = (double *) malloc(sizeof(double) * n * n);
  
	int i, j, k, l;

	//creamos una matriz identidad
	#pragma omp parallel
	{
        #pragma omp for collapse(2) private(j)
	for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                if (i == j) {
                    identidad[i*n+j] = 1;
                }
                else {
                        identidad[i*n+j]=0;
                } 
          }
        }
      	}
    //recorremos los primeros elementos de cada fila para ponerlos a 0 hasta alcanzar el central, la diagonal pondremos a 1           
    for (i = 0; i < n; i++){
    	for (j = 0; j <= i; j++){
        if (i>j){
	        double variable;
	        int indice;
          //buscamos la fila de la matriz que vale 1 en la misma columna
	        indice = j;
	        //calculamos el numero por el que se multiplicara toda la fila
          variable = a[i*n+j]/a[indice*n+j];
          double array_matriz[n], array_identidad[n];
          //restamos la fila multiplicada por el valor de la fila actual
          for (l = 0; l < n; l++){
            array_matriz[l] = a[indice*n+l]*variable;
            a[i*n+l] = a[i*n+l] - array_matriz[l];
            array_identidad[l] = identidad[indice*n+l]*variable;
            identidad[i*n+l] = identidad[i*n+l] - array_identidad[l];
				    }
          }
            //ponemos los elementos centrales en valor 1 y convertimos toda la fila                          
          else if (i == j){
            double variable;
            variable = a[i*n+j];
            for(k = 0; k < n; k++){
            	a[i*n+k] = a[i*n+k]/variable;
			        identidad[i*n+k] = identidad[i*n+k]/variable;
       			}
	    	  }
      }
    }

    //recorremos la matriz desde la esquina inferior derecha para poner a 0 todo a la derecha de la diagonal 
    for (i = (n-1); i >= 0; i--){
    	for (j = (n-1); j >= i; j--){
        if (i<j){
	        double variable;
	        int indice, l;
          //buscamos la fila de la matriz que vale 1 en la misma columna
	        indice = j;
			   	
	     	  //calculamos el numero por el que se multiplicara toda la fila
          variable = a[i*n+j]/a[indice*n+j];
          double array_matriz[n], array_identidad[n];
          //restamos la fila multiplicada por el valor de la fila actual
          for (l = (n-1); l >= 0; l--){
            array_matriz[l] = a[indice*n+l]*variable;
            a[i*n+l] = a[i*n+l] - array_matriz[l];
            array_identidad[l] = identidad[indice*n+l]*variable;
            identidad[i*n+l] = identidad[i*n+l] - array_identidad[l];
				  }
        }
      }
    }


    #pragma omp parallel
    {
    #pragma omp for collapse(2) private(j, k)
    //multiplicamos A^-1 * B para hallar X
    for (i = 0; i < n; i++){
    	for (j = 0; j < n; j++){
	  	    double valor = 0;
		    //suma de fila por columna
		    for(k = 0; k < n; k++){
		    valor = valor + (identidad[i*n+k] * b[k*n+j]);
		    }
		  matrix_x[i*n+j] = valor;
	}
    }
    }

    #pragma omp parallel
    {
    #pragma omp for
    for (i = 0; i < n * n; i++){
      b[i] = matrix_x[i];
    }
    }
    free(identidad);
    free(matrix_x);

}

void main(int argc, char *argv[])
{
  int size = atoi(argv[1]);

  double *a, *aref;
  double *b, *bref;

  a = generate_matrix(size);
  aref = generate_matrix(size);
  b = generate_matrix(size);
  bref = generate_matrix(size);

  // Using LAPACK dgesv OpenBLAS implementation to solve the system
  int n = size, nrhs = size, lda = size, ldb = size, info;
  int *ipiv = (int *) malloc(sizeof(int) * size);

  double tStart = omp_get_wtime();
  info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, aref, lda, ipiv, bref, ldb);
  printf("Time taken by OpenBLAS LAPACK: %.2fs\n", (omp_get_wtime() - tStart));
  int *ipiv2 = (int *) malloc(sizeof(int) * size);
  
  tStart = omp_get_wtime();
  my_dgesv(n, nrhs, a, lda, ipiv2, b, ldb);
  printf("Time taken by my implementation: %.2fs\n", (omp_get_wtime() - tStart));
    
  if (check_result(bref, b, size) == 1)
    printf("Result is ok!\n");
  else
    printf("Result is wrong!\n");

  free(ipiv);
  free(ipiv2);
  free(a);
  free(aref);
  free(b);
  free(bref);
 
}
