#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iterator>
#include <vector>

#define DIM 28
#define NCLASSES 10

#define FIRST_LAYER_NODES 140
#define SECOND_LAYER_NODES 30

#define NPARTS 10

#define BS 1000

#define INERTIA 0.0001
#define PHI_P   1
#define PHI_G   3
#define PI 3.1412

// Layers weights
double w1[NPARTS * FIRST_LAYER_NODES * DIM * DIM];
double w2[NPARTS * FIRST_LAYER_NODES * SECOND_LAYER_NODES];
double wo[NPARTS * SECOND_LAYER_NODES];

// Velocities

double vw1[NPARTS * FIRST_LAYER_NODES * DIM * DIM];
double vw2[NPARTS * FIRST_LAYER_NODES * SECOND_LAYER_NODES];
double vwo[NPARTS * SECOND_LAYER_NODES];

double bw1[FIRST_LAYER_NODES * DIM * DIM];
double bw2[FIRST_LAYER_NODES * SECOND_LAYER_NODES];
double bwo[SECOND_LAYER_NODES];

double gbw1[FIRST_LAYER_NODES * DIM * DIM];
double gbw2[FIRST_LAYER_NODES * SECOND_LAYER_NODES];
double gbwo[SECOND_LAYER_NODES];

// Layers content
double h1[FIRST_LAYER_NODES];
double h2[SECOND_LAYER_NODES];

double sigmoid(double x){
  return exp( x ) / ( 1 + exp(x) );
}

double normal(double u, double mean, double dev) {
  //  return( exp(-0.5 * (u - mean) * (u - mean) / (dev * dev)) * ( 1 / (dev * sqrt(2*PI) ) ) );

  return u / (double) RAND_MAX * 0.000001 - 0.0000005;

}

double bestGLoss = 1E10;

double ComputeBatchLoss(int p, int part, std::vector<unsigned char> labels) {
  double loss = 0;

  int trueLabel = labels[8 + p];
  
  double s = 0;
  for(int j = 0; j < SECOND_LAYER_NODES; j++) {
    s += wo[part * (SECOND_LAYER_NODES) + j] * h2[j];
  }

  s = (s < 0 ? 0: s);

  //  printf("s = %lf, tl = %d\n", s, trueLabel);
  int guess;
  if ( ( s > 0) && ( s <= 1 ) )  guess = 0;
  else
    if ( ( s > 1) && ( s <= 2 ) )  guess = 1;
    else
      if ( ( s > 2) && ( s <= 3 ) )  guess = 2;
      else
	if ( ( s > 3) && ( s <= 4 ) )  guess = 3;
	else
	  if ( ( s > 4) && ( s <= 5 ) )  guess = 4;
	  else
	    if ( ( s > 5) && ( s <= 6 ) )  guess = 5;
	    else
	      if ( ( s > 6) && ( s <= 7 ) )  guess = 6;
	      else
		if ( ( s > 7) && ( s <= 8 ) )  guess = 7;
		else
		  if ( ( s > 8) && ( s <= 9 ) )  guess = 8;
		  else
		    if ( ( s > 9) && ( s <= 10 ) ) guess = 9;
		    else
		      guess = 10;
  
  loss += (guess == trueLabel ? 0 : 1); 
  return loss;
}
double ComputeAcc(int p, std::vector<unsigned char> labels) {
  double loss = 0;
  
  double s = 0;
  for(int j = 0; j < SECOND_LAYER_NODES; j++) {
    s += gbwo[j] * h2[j];
  }

  s = (s < 0 ? 0: s);
  
  int guess;

  if ( ( s > 0) && ( s <= 1 ) )  guess = 0;
  else
    if ( ( s > 1) && ( s <= 2 ) )  guess = 1;
    else
      if ( ( s > 2) && ( s <= 3 ) )  guess = 2;
      else
	if ( ( s > 3) && ( s <= 4 ) )  guess = 3;
	else
	  if ( ( s > 4) && ( s <= 5 ) )  guess = 4;
	  else
	    if ( ( s > 5) && ( s <= 6 ) )  guess = 5;
	    else
	      if ( ( s > 6) && ( s <= 7 ) )  guess = 6;
	      else
		if ( ( s > 7) && ( s <= 8 ) )  guess = 7;
		else
		  if ( ( s > 8) && ( s <= 9 ) )  guess = 8;
		  else
		    if ( ( s > 9) && ( s <= 10 ) ) guess = 9;
		    else
		      guess = 10;
  
  int label = labels[8 + p];
  loss += (guess == label ? 0 : 1); 
  return loss;
}

void WeightsAndVelocitiestRandomization() {
  // weight randomization
  for(int part = 0; part < NPARTS; part++) {
    for(int i = 0; i < FIRST_LAYER_NODES; i++) 
      for(int j = 0; j < DIM * DIM; j++) 
	w1[part * (FIRST_LAYER_NODES * DIM * DIM ) + i * DIM * DIM + j ] = normal(rand(), 0 , sqrt(2/(double)(FIRST_LAYER_NODES * DIM *DIM)));
    for(int i = 0; i < SECOND_LAYER_NODES; i++) 
      for(int j = 0; j < FIRST_LAYER_NODES; j++) 
	w2[part * (FIRST_LAYER_NODES * SECOND_LAYER_NODES) + i * FIRST_LAYER_NODES + j] = normal(rand(), 0 , sqrt(2/(double)(FIRST_LAYER_NODES * SECOND_LAYER_NODES)));
    for(int i = 0; i < NCLASSES; i++) 
      for(int j = 0; j < SECOND_LAYER_NODES; j++) 
	wo[part * SECOND_LAYER_NODES + j] = normal(rand(), 0 , sqrt(2/(double)SECOND_LAYER_NODES));
  }

  // Initialize best global solution
  for(int i = 0; i < FIRST_LAYER_NODES; i++) 
    for(int j = 0; j < DIM * DIM; j++) 
      gbw1[i * DIM * DIM + j] = bw1[i * DIM * DIM + j] = w1[i * DIM * DIM + j];
  for(int i = 0; i < SECOND_LAYER_NODES; i++) 
    for(int j = 0; j < FIRST_LAYER_NODES; j++) 
      gbw2[i * FIRST_LAYER_NODES + j] = bw2[i * FIRST_LAYER_NODES + j] = w2[i * FIRST_LAYER_NODES + j];
  for(int i = 0; i < NCLASSES; i++) 
    for(int j = 0; j < SECOND_LAYER_NODES; j++) 
      gbwo[j] = bwo[j] = wo[j];
  
  // velocities randomization
  for(int part = 0; part < NPARTS; part++) {
    for(int i = 0; i < FIRST_LAYER_NODES; i++) 
      for(int j = 0; j < DIM * DIM; j++) 
	vw1[part * (FIRST_LAYER_NODES * DIM * DIM ) + i * DIM * DIM + j ] = normal(rand(), 0 , sqrt(2/(double)(FIRST_LAYER_NODES * DIM * DIM)));
    for(int i = 0; i < SECOND_LAYER_NODES; i++) 
      for(int j = 0; j < FIRST_LAYER_NODES; j++) 
	vw2[part * (FIRST_LAYER_NODES * SECOND_LAYER_NODES) + i * FIRST_LAYER_NODES + j] = normal(rand(), 0 , sqrt(2/(double)(FIRST_LAYER_NODES * SECOND_LAYER_NODES)));
    for(int i = 0; i < NCLASSES; i++) 
      for(int j = 0; j < SECOND_LAYER_NODES; j++) 
	vwo[part * SECOND_LAYER_NODES + j] = normal(rand(), 0 , sqrt(2/(double)SECOND_LAYER_NODES));
  }
}

int main()
{
  std::ifstream input1( "mnist/train-images-idx3-ubyte", std::ios::binary );
  
  std::vector<unsigned char> train_img(std::istreambuf_iterator<char>(input1), {});

  printf("size of train_img in bytes = %ld\n", train_img.size());

  int nPatterns = (train_img.size() - 16 ) / (DIM * DIM);

  std::ifstream input2( "mnist/train-labels-idx1-ubyte", std::ios::binary );
  
  std::vector<unsigned char> train_lab(std::istreambuf_iterator<char>(input2), {});

  printf("sizeof train_lab in bytes = %ld\n", train_lab.size());

  WeightsAndVelocitiestRandomization();
  
  double best_loss = 1E10;
  double tot_loss[NPARTS];
  for(int part = 0; part < NPARTS; part++) 
    tot_loss[part] = 0;
  int mpart = 0;

  int nbs = nPatterns / BS;
  int epoch = 0;
  int nIterNoImprove = 0;
  int nIter = 0;
  
  while(1) {
    for(int b = 0; b < nbs; b++) {
      for(int part = 0; part < NPARTS; part++) {
	tot_loss[part] = 0;
	for (int p = b * BS; p < b * BS + BS - 1; p++) {    
	  // First layer
	  for(int i = 0; i < FIRST_LAYER_NODES; i++) {
	    h1[i] = 0;
	    for(int j = 0; j < DIM * DIM; j++) {
	      h1[i] += train_img[16 + p * DIM * DIM + j] / (double) 255 * w1[part * (FIRST_LAYER_NODES * DIM * DIM ) + i * DIM * DIM + j];
	    }
	    h1[i] = (h1[i] < 0 ? 0: h1[i]);
	  }
    
	  // Second layer
	  for(int i = 0; i < SECOND_LAYER_NODES; i++) {
	    h2[i] = 0;
	    for(int j = 0; j < FIRST_LAYER_NODES; j++) {
	      h2[i] += h1[j] * w2[part * (FIRST_LAYER_NODES * SECOND_LAYER_NODES) + i * FIRST_LAYER_NODES + j];
	    }
	    h2[i] = (h2[i] < 0 ? 0: h2[i]);
	    //	    h2[i] = sigmoid(h2[i]);
	  }
	  tot_loss[part] += ComputeBatchLoss(p, part, train_lab);
	  //	  printf("%lf %lf %lf\n", ComputeBatchLoss(p, 0, train_lab), ComputeBatchLoss(p, 1, train_lab), ComputeBatchLoss(p, 2, train_lab));
	}
      }

      if (bestGLoss == 1E10)
	bestGLoss = tot_loss[0];
      
      double mloss = 1E10;
	    
      for(int i = 0; i < NPARTS; i++) {
	//	printf("tot_loss[%d] = %lf\n", i, tot_loss[i]);
	if (tot_loss[i] < mloss) {
	  mpart = i;
	  mloss = tot_loss[i];
	}
      }
      printf("BEST LOSS = %lf\n", mloss);

      double r_p = rand() / (double) RAND_MAX;
      double r_g = rand() / (double) RAND_MAX;
      int index;

      for(int parti = 0; parti < NPARTS; parti++) {
	  // UPDATE VELOCITIES
	  for(int i = 0; i < FIRST_LAYER_NODES; i++) 
	    for(int j = 0; j < DIM * DIM; j++) { 
	      index = parti * (FIRST_LAYER_NODES * DIM * DIM ) + j + i * DIM * DIM;
	      vw1[index] = INERTIA * vw1[index] + PHI_P * r_p * (bw1[i * DIM * DIM + j] - w1[index]) + PHI_G * r_g * (gbw1[i * DIM * DIM + j] - w1[index]);
	    }
	  for(int i = 0; i < SECOND_LAYER_NODES; i++) 
	    for(int j = 0; j < FIRST_LAYER_NODES; j++) {
	      index = parti * (FIRST_LAYER_NODES * SECOND_LAYER_NODES) + j + i * FIRST_LAYER_NODES;
	      vw2[index] = INERTIA * vw2[index] + PHI_P * r_p * (bw2[i * FIRST_LAYER_NODES +j] - w2[index]) + PHI_G * r_g * (gbw2[i * FIRST_LAYER_NODES +j] - w2[index]);
	    }
	  for(int j = 0; j < SECOND_LAYER_NODES; j++) {
	    index = parti * (SECOND_LAYER_NODES) + j;
	    vwo[index] = INERTIA * vwo[index] + PHI_P * r_p * (bwo[j] - wo[index]) + PHI_G * r_g * (gbwo[j] - wo[index]);
	  }
      }
      
      // UPDATE POSITIONS
      
      for(int parti = 0; parti < NPARTS; parti++) {
	for(int i = 0; i < FIRST_LAYER_NODES; i++) 
	    for(int j = 0; j < DIM * DIM; j++) {
	      index = parti * (FIRST_LAYER_NODES * DIM * DIM ) + j + i * DIM * DIM;
	      w1[index] += vw1[index] + normal(rand(), 0 , 1);
	    }
	  for(int i = 0; i < SECOND_LAYER_NODES; i++) 
	    for(int j = 0; j < FIRST_LAYER_NODES; j++) {
	      index = parti * (FIRST_LAYER_NODES * SECOND_LAYER_NODES) + j + i * FIRST_LAYER_NODES;
	      w2[index] += vw2[index]+ normal(rand(), 0 , 1);;
	    }
	  for(int j = 0; j < SECOND_LAYER_NODES; j++) {
	    index = parti * (SECOND_LAYER_NODES) + j; 
	    wo[index] += vwo[index]+ normal(rand(), 0 , 1);;
	  }
      }
      
      if (mloss < best_loss) {
	nIterNoImprove=0;
	printf("IMPROVED!!!!!!!!!!!!!!!!!!!!!!!!!\n");
	
	// Set best weights
	
	for(int i = 0; i < FIRST_LAYER_NODES * DIM * DIM; i++) 
	  bw1[i] = w1[mpart * (FIRST_LAYER_NODES * DIM * DIM ) + i];
	
	for(int i = 0; i < FIRST_LAYER_NODES * SECOND_LAYER_NODES; i++) 
	  bw2[i] = w2[mpart * (FIRST_LAYER_NODES * SECOND_LAYER_NODES) + i];
	
	for(int i = 0; i < SECOND_LAYER_NODES; i++) 
	  bwo[i] = wo[mpart * (SECOND_LAYER_NODES) + i];
	
	best_loss = mloss;

	if (mloss < bestGLoss) {
	  // Initialize best global solution
	  for(int i = 0; i < FIRST_LAYER_NODES * DIM * DIM; i++)
	    gbw1[i] = bw1[i];
	  
	  for(int i = 0; i < FIRST_LAYER_NODES * SECOND_LAYER_NODES; i++) 
	    gbw2[i] = bw2[i];
	  
	  for(int i = 0; i < SECOND_LAYER_NODES; i++) 
	    gbwo[i] = bwo[i];
	  
	  bestGLoss = mloss;
	}
	else {
	  nIterNoImprove++;
	  if (nIterNoImprove == 20) {
	    nIterNoImprove = 0;
	      // weight randomization
	    for(int i = 0; i < FIRST_LAYER_NODES; i++) 
	      for(int j = 0; j < DIM * DIM; j++) 
		w1[ i * DIM * DIM + j ] = gbw1[i * DIM * DIM + j];
	    for(int i = 0; i < SECOND_LAYER_NODES; i++) 
	      for(int j = 0; j < FIRST_LAYER_NODES; j++) 
		w2[ i * FIRST_LAYER_NODES + j] = gbw2[i * FIRST_LAYER_NODES + j];
	    for(int i = 0; i < NCLASSES; i++) 
	      for(int j = 0; j < SECOND_LAYER_NODES; j++) 
		wo[ j] = gbwo[j];

	    for(int part = 1; part < NPARTS; part++) {
	      for(int i = 0; i < FIRST_LAYER_NODES; i++) 
		for(int j = 0; j < DIM * DIM; j++) 
		  w1[part * (FIRST_LAYER_NODES * DIM * DIM ) + i * DIM * DIM + j ] = normal(rand(), 0 , sqrt(2/(double)(FIRST_LAYER_NODES * DIM *DIM)));
	      for(int i = 0; i < SECOND_LAYER_NODES; i++) 
		for(int j = 0; j < FIRST_LAYER_NODES; j++) 
		  w2[part * (FIRST_LAYER_NODES * SECOND_LAYER_NODES) + i * FIRST_LAYER_NODES + j] = normal(rand(), 0 , sqrt(2/(double)(FIRST_LAYER_NODES * SECOND_LAYER_NODES)));
	      for(int i = 0; i < NCLASSES; i++) 
		for(int j = 0; j < SECOND_LAYER_NODES; j++) 
		  wo[part * SECOND_LAYER_NODES + j] = normal(rand(), 0 , sqrt(2/(double)SECOND_LAYER_NODES));
	    }
	  }
	}
      }
    }
    
    double acc = 0;
    for(int p = 0; p < nPatterns; p++) {
      // First layer
      for(int i = 0; i < FIRST_LAYER_NODES; i++) {
	h1[i] = 0;
	for(int j = 0; j < DIM * DIM; j++) {
	  h1[i] += train_img[16 + j + p * DIM * DIM] / (double) 255 * gbw1[j + i * DIM * DIM];
	}
	h1[i] = (h1[i] < 0 ? 0: h1[i]);
      }
    
      // Second layer
      for(int i = 0; i < SECOND_LAYER_NODES; i++) {
	h2[i] = 0;
	for(int j = 0; j < FIRST_LAYER_NODES; j++) {
	  h2[i] += h1[j] * gbw2[j + i * FIRST_LAYER_NODES];
	}
	h2[i] = (h2[i] < 0 ? 0: h2[i]);
	// h2[i] = sigmoid(h2[i]);
      }
      
      acc += ComputeAcc(p, train_lab);      
    }
    printf("EPOCH: %d, accuracy = %lf\n", epoch, 1 - acc/(double)nPatterns);
    epoch++;
  }  
  return 0;
}

