#include "mex.h"
#include "cokus.cpp"
#include <list>
#include <iterator> 
using namespace std;

// with a reference source from Mark Steyvers topic modeling toolbox
// http://psiexp.ss.uci.edu/research/programs_data/toolbox.htm
 
// constraint LDA 
void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
                 const mxArray *prhs[])
{
  mxArray  *cell_element_ptr;
  double *srwp, *srdp, *probs, *Z, *WS, *DS, *ZIN, *tokenIdx, *nFC, *cellData;
  double ALPHA,BETA;
  mwIndex *irwp, *jcwp, *irdp, *jcdp;
  int *z,*d,*w, *order, *wp, *dp, *ztot, *idx, *fc;
  int W,T,D,NN,SEED,OUTPUT, nzmax, nzmaxwp, nzmaxdp, ntokens, nFea;
  int i,ii,j,c,n,nt,wi,di, startcond;
  
  startcond = 0;  
  // pointer to word indices
  WS = mxGetPr( prhs[ 0 ] );
  // pointer to document indices
  DS = mxGetPr( prhs[ 1 ] );
  T = (int) mxGetScalar(prhs[2]);
  NN = (int) mxGetScalar(prhs[3]);
  ALPHA = (double) mxGetScalar(prhs[4]);  
  BETA = (double) mxGetScalar(prhs[5]);  
  SEED = (int) mxGetScalar(prhs[6]);  
  OUTPUT = (int) mxGetScalar(prhs[7]);
  tokenIdx = mxGetPr(prhs[8]);
  nFC = mxGetPr(prhs[9]);
  
  // get the number of tokens
  ntokens = mxGetM( prhs[ 0 ] ) * mxGetN( prhs[ 0 ] );
  
  // seeding
  seedMT( 1 + SEED * 2 ); // seeding only works on uneven numbers

  /* allocate memory */
  z  = (int *) mxCalloc( ntokens , sizeof( int ));
  
  if (startcond == 1) {
     for (i=0; i<ntokens; i++) 
         z[ i ] = (int) ZIN[ i ] - 1;   
  }
  
  d  = (int *) mxCalloc( ntokens , sizeof( int ));
  w  = (int *) mxCalloc( ntokens , sizeof( int ));
  idx  = (int *) mxCalloc( ntokens , sizeof( int ));
  
  order  = (int *) mxCalloc( ntokens , sizeof( int ));  
  ztot  = (int *) mxCalloc( T , sizeof( int ));
  probs  = (double *) mxCalloc( T , sizeof( double ));
  
  // copy over the word and document indices into internal format
  for (i=0; i<ntokens; i++) {
     w[ i ] = (int) WS[ i ] - 1;
     d[ i ] = (int) DS[ i ] - 1;
     idx[ i ] = (int) tokenIdx[ i ] - 1;
  }
  
  n = ntokens;
  
  W = 0;
  D = 0;
  for (i=0; i<n; i++) {
     if (w[ i ] > W) W = w[ i ];
     if (d[ i ] > D) D = d[ i ];
  }
  W = W + 1;
  D = D + 1;
  
  nFea = n/D;
//   mexPrintf( "number of features is %d\n", nFea);
  fc = (int *) mxCalloc( nFea , sizeof( int ));
  
  for (i = 0; i<nFea; i++){
      fc[i] = nFC[i]; 
  }
  
  wp = (int *) mxCalloc(T*W , sizeof(int));
  dp = (int *) mxCalloc(T*D , sizeof(int));
   

  /* run the model */
  
  int iii,topic, rp, temp, iter, wioffset, dioffset;
  double totprob, WBETA, r, max, phi_hat, theta_hat, scalar;
  
  if (startcond == 0) {
  /* random initialization */
      if (OUTPUT == 2) mexPrintf( "Starting Random initialization\n" );
      for (i = 0; i< n; i++)
      {
          wi = w[ i ];
          di = d[ i ];
          // pick a random topic 0..T-1
          topic = (int) ( (double) randomMT() * (double) T / (double) (4294967296.0 + 1.0) );
          z[i] = topic; // assign this word token to this topic
          wp[ wi*T + topic ]++; // increment wp count matrix
          dp[ di*T + topic ]++; // increment dp count matrix
          ztot[ topic ]++; // increment ztot matrix
      }
  }
  
  for (i = 0; i < n; i++) order[i]=i; // fill with increasing series
  for (i = 0; i < (n-1); i++) {
      // pick a random integer between i and nw
      rp = i + (int) ((double) (n-i) * (double) randomMT() / (double) (4294967296.0 + 1.0));
      // switch contents on position i and position rp
      temp = order[rp];
      order[rp]=order[i];
      order[i]=temp;
  }
  
  list<int>::iterator it;
//   list<int> feaIndex;
  list<int> feaIndexWords;  
  
  WBETA = (double) (W*BETA);
  for (iter=0; iter<NN; iter++) {

      for (ii = 0; ii < n; ii++) {
          i = order[ ii ]; // current word token to assess
          wi  = w[i]; // current word index
          di  = d[i]; // current document index  
          topic = z[i]; // current topic assignment to word token
          ztot[topic]--;  // substract this from counts
          
          wioffset = wi*T;
          dioffset = di*T;
          
          wp[wioffset+topic]--;
          dp[dioffset+topic]--;

          int reminder, toPush, Nb, indexF;
          reminder = i%nFea;
          indexF = idx[i]; // nth feature
          Nb = fc[indexF]; // nb. of feature values (words)

          cell_element_ptr = mxGetCell(prhs[10], indexF);
          cellData = mxGetPr(cell_element_ptr);

          for (iii = 0; iii < Nb; iii++){
              feaIndexWords.push_back((int)(cellData[iii]-1));
          }
          
          /*remove the current studied word in the words list */
          for (it = feaIndexWords.begin(); it != feaIndexWords.end(); it++){
              if (*it == wi){
                  feaIndexWords.erase(it); // remove the current word from list
                  break;
              }
          }
          
          totprob = (double) 0;
          for (j = 0; j < T; j++) {
              int tf = 0;
              for (it = feaIndexWords.begin(); it != feaIndexWords.end(); it++){
                  tf = tf + (int)wp[*it*T+j];
              }
              if (tf == 0 && wp[ wioffset+j ] == 0){
                  scalar = (double)1;
              }
              if (tf == 0 && wp[ wioffset+j ] != 0){
                  scalar = (double)1;
              }              
              if (tf != 0 && wp[ wioffset+j ] == 0){
                  scalar = (double)1/((double)tf+(double)feaIndexWords.size());
              }
              if (tf != 0 && wp[ wioffset+j ] != 0){
                  scalar = (double)wp[ wioffset+j ]/((double)tf+(double)wp[ wioffset+j ]);
              }              
             
              phi_hat = ((double)wp[wioffset+j] + (double)BETA)/( (double)ztot[j] + (double)WBETA);
              theta_hat = ((double)dp[dioffset+j] + (double)ALPHA);
              probs[j] = phi_hat*theta_hat*scalar;
              totprob += probs[j];
          }
          
          // sample a topic from the distribution
          r = (double) totprob * (double) randomMT() / (double) 4294967296.0;
          max = probs[0];
          topic = 0;
          while (r > max) {
              topic++;
              max += probs[topic];
          }
           
          z[i] = topic; // assign current word token i to topic j
          wp[wioffset + topic ]++; // and update counts
          dp[dioffset + topic ]++;
          ztot[topic]++;
          feaIndexWords.clear(); // clear words content
          
      }
  }
  
  /* convert the full wp matrix into a sparse matrix */
  nzmaxwp = 0;
  for (i = 0; i < W; i++) {
     for (j = 0; j < T; j++)
         nzmaxwp += (int) ( *( wp + j + i*T )) > 0;
  }  

  
  // MAKE THE WP SPARSE MATRIX
  plhs[0] = mxCreateSparse( W,T,nzmaxwp,mxREAL);
  srwp  = mxGetPr(plhs[0]);
  irwp = mxGetIr(plhs[0]);
  jcwp = mxGetJc(plhs[0]);  
  n = 0;
  for (j = 0; j<T; j++) {
      *( jcwp + j ) = n;
      for (i = 0; i<W; i++) {
         c = (int) *( wp + i*T + j );
         if (c >0) {
             *( srwp + n ) = c;
             *( irwp + n ) = i;
             n++;
         }
      }    
  }  
  *( jcwp + T ) = n;    
   
  // MAKE THE DP SPARSE MATRIX
  nzmaxdp = 0;
  for (i = 0; i < D; i++) {
      for (j = 0; j < T; j++)
          nzmaxdp += (int) ( *( dp + j + i*T )) > 0;
  }  

  plhs[1] = mxCreateSparse( D,T,nzmaxdp,mxREAL);
  srdp  = mxGetPr(plhs[1]);
  irdp = mxGetIr(plhs[1]);
  jcdp = mxGetJc(plhs[1]);
  n = 0;
  for (j = 0; j < T; j++) {
      *( jcdp + j ) = n;
      for (i = 0; i < D; i++) {
          c = (int) *( dp + i*T + j );
          if (c >0) {
              *( srdp + n ) = c;
              *( irdp + n ) = i;
              n++;
          }
      }
  }
  *( jcdp + T ) = n;
  
  plhs[ 2 ] = mxCreateDoubleMatrix( 1,ntokens , mxREAL );
  Z = mxGetPr( plhs[ 2 ] );
  for (i=0; i<ntokens; i++) Z[ i ] = (double) z[ i ] + 1;

}


