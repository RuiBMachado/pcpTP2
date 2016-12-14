/****************************************************************************
 * DESCRIPTION:  
 *   MPI HEAT2D Example - C Version
 *   This example is based on a simplified 
 *   two-dimensional heat equation domain decomposition.  The initial 
 *   temperature is computed to be high in the middle of the domain and 
 *   zero at the boundaries.  The boundaries are held at zero throughout 
 *   the simulation.  During the time-stepping, an array containing two 
 *   domains is used; these domains alternate between old data and new data.
 *
 *  The physical region, and the boundary conditions, are suggested
    by this diagram;
                   u = 0
             +------------------+
             |                  |
    u = 100  |                  | u = 100
             |                  |
             |                  |
             |                  |
             |                  |
             +------------------+
                   u = 100
Interrior point :
  u[Central] = (1/4) * ( u[North] + u[South] + u[East] + u[West] )
PARALLEL MPI VERSION :
           +-------------------+
           |                   | P0   m=(n-2)/P +2
           +-------------------+
           |                   | P1
           +-------------------+
       n   |                   | ..
           +-------------------+
           |                   | Pq
           +-------------------+
                 
           <-------- n -------->  
            <-------n-2 ------>
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#define NN 200
#define MM 200  

double update(int rank,int size, int nx,int ny, double *u, double *unew);
void inicializa(int rank, int size, int nx, int ny, double *u); 
void imprime(int rank, int nx, int ny, double *u,const char *fnam);




int main(int argc, char *argv[])
{ 
    int N=NN,M=MM;
    float epsilon;
    int qtdlinhas;
    int size,rank,i;

   
    /* INITIALIZE MPI */
    MPI_Init(&argc, &argv);

    /* GET THE PROCESSOR ID AND NUMBER OF PROCESSORS */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //Only Rank 0 read application parameters
    if(rank==0) {     
        epsilon = atof(argv[1]);
    }

    //Wait for rank 0 , all process start here 
    MPI_Barrier(MPI_COMM_WORLD);

    //Exchange M
    MPI_Bcast(&M , 1, MPI_INT, 0 , MPI_COMM_WORLD);
    //Exchange epsilon  
    MPI_Bcast(&epsilon , 1, MPI_FLOAT, 0 , MPI_COMM_WORLD);
    MPI_Status status;

   if(rank==0){

    int linhas = NN/size+2;
    int j;
    int extra = NN%size;
    int *processos    = (int*)malloc(size * sizeof(int));
  
  for (i=0; i<size; i++)
      {
           if(i<extra){
            processos[i]=linhas+1;
        }
           else{
            processos[i]=linhas;

      }
    }
      for(j=1;j<size;j++)
        MPI_Send(&processos[j],1, MPI_INT,j, 0,MPI_COMM_WORLD); 
    
    if(processos[0]==1)
          qtdlinhas=processos[0];
    else qtdlinhas = processos[0]-1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    if(rank>0) {
      if(rank==size-1){
                MPI_Recv(&qtdlinhas,1,MPI_INT,0,0,MPI_COMM_WORLD, &status);
                if(qtdlinhas>1)qtdlinhas=qtdlinhas-1;

      }else
        MPI_Recv(&qtdlinhas,1,MPI_INT,0,0,MPI_COMM_WORLD, &status);
    }
     //   M=(NN)/size + 2;
    //local size
   //if(rank==0)M=M-1;
  //if(rank==size-1) M = M-1;

    MPI_Barrier(MPI_COMM_WORLD);

    double *u     = (double *)malloc(qtdlinhas * M * sizeof(double));
    double *unew  = (double *)malloc(qtdlinhas * M * sizeof(double));

    /* Initialize grid and create input file 
     * each process initialize its part
     * */
    inicializa(rank,size,qtdlinhas,M,u);
//         imprime(rank,M,N, u, "inicial.txt");

    if (rank == 0) {

        printf ( "\n" );
        printf ( " Iteration  Change\n" );
        printf ( "\n" );
     } 

    double diff, globaldiff=1.0;
    int iterations = 0;
    int iterations_print = 1;
    double start = MPI_Wtime(); //inicio contagem do tempo

    /*
     *   iterate (JACOBI ITERATION) until the new solution unew differs from the old solution u
     *     by no more than epsilon.
     **/

    while( epsilon<=globaldiff )  {

        diff= update(rank,size,qtdlinhas,M, u, unew);

        /**
         *   COMPUTE GLOBAL CONVERGENCE
         * */

        MPI_Allreduce(&diff, &globaldiff , 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); 
        
        if(rank==0){
            iterations++;

                if ( iterations == iterations_print )
                {
                  printf ( "  %8d  %f\n", iterations, globaldiff );
                  iterations_print = 2 * iterations_print;
                }
        }
    }
     double end = MPI_Wtime();  //fim da contagem do tempo
 
    if (rank == 0) {
        
        printf ( "\n" );
        printf ( "  %8d  %f\n", iterations, globaldiff );
        printf ( "\n" );
        printf ( "  Error tolerance achieved.\n" );
        printf("Concluido com %d processos em %f segundos.\n", size, (end-start));      
        printf ( "\n" );
        printf ("  Solution written to the output file %s\n", "final.txt" );
        printf ( "  Normal end of execution.\n" );
        

        imprime(rank,qtdlinhas-2,M, u, "final.txt");
        
        for (int i = 1; i < size; i++) {

          MPI_Recv(&qtdlinhas,1,MPI_INT,i,0,MPI_COMM_WORLD, &status);

          double *buffer    = (double *)malloc(qtdlinhas * M * sizeof(double));

          MPI_Recv(buffer,qtdlinhas*M,MPI_DOUBLE,i,0,MPI_COMM_WORLD, &status);
          if(i==size-1)imprime(i,qtdlinhas,M, buffer, "final.txt");
          else imprime(i,qtdlinhas-2,M, buffer, "final.txt");
          free(buffer);
      
      }

     }else {
          MPI_Send(&qtdlinhas,1, MPI_INT,0, 0,MPI_COMM_WORLD);
          MPI_Send(u,qtdlinhas*M, MPI_DOUBLE,0, 0,MPI_COMM_WORLD);
          }

    free(u);
    free(unew);
    MPI_Finalize();
}



/****************************************************************************
 *  subroutine update
 ****************************************************************************/
double update(int rank, int size, int nx,int ny, double *u, double *unew){
    int ix, iy;
    double  diff=0.0;
    MPI_Status status;

    /*
     * EXCHANGE GHOST CELL
     */
    if (rank > 0 && rank< size-1)
    {
        MPI_Sendrecv(&u[ny*(nx-2)], ny, MPI_DOUBLE, rank+1, 0,
                &u[ny*0],     ny, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&u[ny*1],     ny, MPI_DOUBLE, rank-1, 1,
                &u[ny*(nx-1)], ny, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &status);
    }

    else if (rank == 0)
        MPI_Sendrecv(&u[ny*(nx-2)], ny, MPI_DOUBLE, rank+1, 0,
                &u[ny*(nx-1)], ny, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &status);
    else if (rank == size-1)
        MPI_Sendrecv(&u[ny*1],     ny, MPI_DOUBLE, rank-1, 1,
                &u[ny*0],     ny, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &status);



    /**
     * PERFORM LOCAL COMPUTATION
     * */

    for (ix = 1; ix < nx-1; ix++) {
        for (iy = 1; iy < ny-1; iy++) {
            unew[ix*ny+iy] = (u[(ix+1)*ny+iy] +  u[(ix-1)*ny+iy] + u[ix*ny+iy+1] +  u[ix*ny+iy-1] )/4.0;
           if(diff< fabs(unew[ix*ny+iy] - u[ix*ny+iy]))
              diff=fabs(unew[ix*ny+iy] - u[ix*ny+iy]);
        }

    }


    for (ix = 1; ix < nx-1; ix++) {
        for (iy = 1; iy < ny-1; iy++) {
            u[ix*ny+iy] = unew[ix*ny+iy]; 
        }
    }  


    return diff;   
}

/*****************************************************************************
 *  Initialize Data
 *****************************************************************************/

void inicializa(int rank, int size,int nx, int ny, double *u) 
{
    int ix, iy;

    /*
     *Set boundary data and interrior values
     * */


    // interior points
    for (ix = 1; ix < nx-1; ix++) 
        for (iy = 1; iy < ny-1; iy++) { 
            u[ix*ny+iy]=0.0; 
        }

    //boundary left
    for (ix = 0; ix < nx; ix++){ 
        u[ix*ny]=0.0; 

    }

    //boundary right
    for (ix = 0; ix < nx; ix++){ 
        u[ix*ny+ (ny-1)]=0.0; 

    }

    //boundary down
    for (iy = 0; iy < ny; iy++){ 

        if(rank==size-1) {
            u[(nx-1)*(ny)+iy]=100.0; 

        }else
        {
            u[(nx-1)*(ny)+iy]=0.0;
        }
    }

    //boundary top
    for (iy = 1; iy < ny; iy++){ 
        u[iy]=0.0;
    }
}


/***************************************************************************
 * Print Data to file
 **************************************************************************/

void imprime(int rank, int nx, int ny, double *u,const char *fname)
{
    int ix, iy;
    FILE *fp;

    fp = fopen(fname, "a");

    if(rank==0) {
        fprintf(fp,"%d", NN);
        fputc ( '\n', fp);
        fprintf(fp, "%d",MM);
        fputc ( '\n', fp);

    }
    for (ix = 0 ; ix < nx; ix++) {
        for (iy =0; iy < ny; iy++) {

            fprintf(fp, "%6.2f ", u[ix*ny+iy]);
        }
       // fprintf(fp, "RANK%d\n",rank );
        fputc ( '\n', fp);
    }

    fclose(fp);
}
