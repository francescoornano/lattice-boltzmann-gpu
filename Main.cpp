#include<iostream>
#include<stdlib.h> 
#include<cmath>
#include<cstring>
#include<cstdlib>
#include<fstream>
#include<sstream>
#include<string>

#include<omp.h>

using namespace std;

int main() {

    double start = omp_get_wtime();
    

    // lattice geometry 
    const size_t m {512};
    const size_t n {128};
    // D2Q9 distribution dimensions
    const int qDim = 9;
    const int dDim = 2;
 
    // object geometry 
    constexpr float rad {11.f}; //radius of solid particle
    int objx = static_cast<int>(m/5);
    int objy = static_cast<int>(n/2+2);
    int objx2 = static_cast<int>(m/3);
    int objy2 = static_cast<int>(n/2+2);

    // kinematic viscosity calculation
    constexpr float uMax {0.3f};
    constexpr float Re {100};
    constexpr float nu = uMax * 2. * rad / Re;
    constexpr float omega = 1. / (3. * nu + 1./2.);
    // TRT parameters 
    const float magicParam {1./4.f};
    //const double tau_s = nu / (pow((1.f / sqrt(3.f)),2));
    const float tau_s = omega;
    const float tau_a = magicParam / (tau_s - 0.5f) + 0.5f;
 
    // simulation parameters 
    const int minIt {5100}; // minimum number of time steps
    const int timePlot {1000}; //plotting time
    const bool outputFlag {1};

    int i_p, j_p; // periodic indices
    
    const float rhoS {1.0}; // initial fluid density
    
    // int array to store walls 
    int isSolid[m * n];
    
    // velocity array
    float u[m* n * dDim];

    // density array
    float rho[m * n];
    
    // post-streaming distribution function
    float fIn[m * n * qDim];

    //pre-streaming distribution function
    float fOut[m * n * qDim];
    
    // equilibrium distribution function
    float fEq[m * n * qDim];
    
    // TRT variables
    float fIn_s, fIn_a, fEq_s, fEq_a, omega_trt;

    // lattice D2Q9 constants 
    const float t[qDim] = {4./9., 1./9., 1./9.,
                                 1./9., 1./9.,
                                  1./36., 1./36.,
                                 1./36., 1./36.}; //t: lattice weights

    // discrete velocity directions array
    const int e[qDim * dDim] = {0,1,0,-1,0,1,-1,-1,1,0,0,1,0,-1,1,1,-1,-1};
    
    //array of opposite discrete velocity directions
    int opp[qDim] = {0,3,4,1,2,7,8,5,6};
   
    // arrays useful for inlet BC
    int col[n-2];
    float y_phys[n-2];
    
    float cu;
    int i, j, k;
    
     // copy arrays to device 
     #pragma omp target enter data map(to: fIn, fOut, fEq, u, rho, isSolid, e, opp, t, col, y_phys) device(0)
     
  
    //Build lattice geometry
    #pragma omp target teams distribute parallel for private(i,j) collapse(2)
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
             
             if ( pow((i - objx),2) + pow((j - objy),2) <= pow(rad,2) )
             {
                   isSolid[i + j*m] = 1;
             }
                       
             else if ( pow((i - objx2),2) + pow((j - objy2),2) <= pow(rad,2) )
             {
                   isSolid[i + j*m] = 1;
             }
                         
             else if ( j == n-1  )
             {
                   isSolid[i + j*m] = 1;
             }
        }
    }


    //Initial conditions
    #pragma omp target teams distribute parallel for private(i,j) collapse(2)
    for (i = 0; i < m; i++) 
    {
        for (j = 0; j < n; j++)
        {
	    u[i*dDim + j*m*dDim]     = uMax;
	    u[1 + i*dDim + j*m*dDim] = 0.;
	    rho[i + j*m]             = rhoS;
        }
    }

    #pragma omp target teams distribute parallel for private(j,k,cu) collapse(2)
    for (i = 0; i < m; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            for (k = 0; k < qDim; ++k)
            {
                 cu = 3.*(static_cast<float>(e[k])*u[i*dDim + j*m*dDim] 
                      + static_cast<float>(e[k + qDim])*u[1 + i*dDim + j*m*dDim]);
                 //fIn(i,j,k) = t.at(k) * rho.at(i + j*m)*(1. + cu + 0.5*(cu*cu)
                 //             - 3./2.*(pow(u.at(i*dDim + j*m*dDim),2) + pow(u.at(1 + i*dDim + j*m*dDim),2)));//popul. distrib. function
                 //populate distribution function
                 fIn[k + i*qDim + j*m*qDim] = t[k] * rho[i + j*m]*(1. + cu + 0.5*(cu*cu)  
                                                 - 3./2.*(pow(u[i*dDim + j*m*dDim],2) 
                                                 + pow(u[1 + i*dDim + j*m*dDim],2)));
            }
        }
    }
    
    //Initialise useful variables
    float L {static_cast<float>(n) - 2.};
   //Following loop is run in serial 
    for (size_t l = 0; l < L; ++l)
    {
        col[l] = l + 1;
        y_phys[l] = static_cast<float>(col[l] - 0.5f);
    }
    
    // update col and y_phys to device memory 
    #pragma omp target update to(col, y_phys) device(0)
    

// Time loop
for (int simTime = 0; simTime < minIt; simTime++)
{

  int i, j;
  int idx;
      
    // Macroscopic Dirichlet BCs
    #pragma omp target teams distribute parallel for private(idx) device(0)
    for (i = 0; i < n-2; ++i)
    {
       idx = col[i];
       
       // Inlet: Pousille BC
       u[idx*m*dDim] = 4. * uMax / (L*L) * (y_phys[i]*L - y_phys[i]*y_phys[i]);
       u[1 + idx*m*dDim] = 0.; 
       rho[idx*m] = 1. / (1. - u[idx*m*dDim]) * (fIn[idx*m*qDim]  
                       + fIn[2 + idx*m*qDim] 
                       + fIn[4 + idx*m*qDim] 
                       + 2. * (fIn[3 + idx*m*qDim]  
                       + fIn[6 + idx*m*qDim] 
                       + fIn[7 + idx*m*qDim] ));
                       
       // Outlet: constant pressure BC
       rho[m-1 + idx*m] = 1.;
       u[(m-1)*dDim + idx*m*dDim] = -1.+1./(rho[m-1 + idx*m])*( fIn[(m-1)*qDim + idx*m*qDim] 
                                       + fIn[2 + (m-1)*qDim + idx*m*qDim] 
                                       + fIn[4 + (m-1)*qDim + idx*m*qDim] 
                                       + 2. *( fIn[1 + (m-1)*qDim + idx*m*qDim]
                                       + fIn[5 + (m-1)*qDim + idx*m*qDim]
                                       + fIn[8 + (m-1)*qDim + idx*m*qDim] ));
       u[1 + (m-1)*dDim + idx*m*dDim] = 0.;
    }
    
    
    // Microscopic BCs
    #pragma omp target teams distribute parallel for private (idx)
    for (i = 0; i < n-2; ++i)
    {
        // Inlet: Zou/He BC
        idx = col[i];
        fIn[1 + idx*m*qDim] = fIn[3 + idx*m*qDim] + 2./3.*rho[idx*m]*u[idx*m*dDim];
        fIn[5 + idx*m*qDim] = fIn[7 + idx*m*qDim] + 1./2.*(fIn[4 + idx*m*qDim] - fIn[2 + idx*m*qDim] )
                                    + 1./2.*rho[idx*m]*u[1 + idx*m*dDim]
                                    + 1./6.*rho[idx*m]*u[idx*m*dDim];
        fIn[8 + idx*m*qDim] = fIn[6 + idx*m*qDim] + 1./2.*(fIn[2 + idx*m*qDim] - fIn[4 + idx*m*qDim] )
                                    - 1./2.*rho[idx*m]*u[1 + idx*m*dDim]
                                    + 1./6.*rho[idx*m]*u[idx*m*dDim];

        // Outlet: Zou/He BC
        fIn[3 + (m-1)*qDim + idx*m*qDim] = fIn[1 + (m-1)*qDim + idx*m*qDim] 
                                              - 2./3.*rho[m-1 + idx*m]*u[(m-1)*dDim + idx*m*dDim];
        fIn[7 + (m-1)*qDim + idx*m*qDim] = fIn[5 + (m-1)*qDim + idx*m*qDim] 
                                              + 1./2.*(fIn[2 + (m-1)*qDim + idx*m*qDim] 
                                              - fIn[4 + (m-1)*qDim + idx*m*qDim] )
                                              - 1./2.*rho[m-1 + idx*m] * u[1 + (m-1)*dDim + idx*m*dDim]
                                              - 1./6.*rho[m-1 + idx*m] * u[(m-1)*dDim + idx*m*dDim];
        fIn[6 + (m-1)*qDim + idx*m*qDim] = fIn[8 + (m-1)*qDim + idx*m*qDim] 
                                              + 1./2.*(fIn[4 + (m-1)*qDim + idx*m*qDim] 
                                              - fIn[2 + (m-1)*qDim + idx*m*qDim] )
                                              + 1./2.*rho[m-1 + idx*m] * u[1 + (m-1)*dDim + idx*m*dDim]
                                              - 1./6.*rho[m-1 + idx*m] * u[(m-1)*dDim + idx*m*dDim];
    }
    
    
    //Compute macroscopic density and velocity 
    #pragma omp target teams distribute parallel for private(j) collapse(2) device(0)
    for (i = 0; i < m; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            u[i*dDim + j*m*dDim] = 0.0;
            u[1 + i*dDim + j*m*dDim] = 0.0;
            rho[i + j*m] = 0.0;

            if (isSolid[i + j*m] == 0)
            {
                for (int k = 0; k < qDim; ++k)
                {
                    //macroscopic density
                    rho[i + j*m] = rho[i + j*m] + fIn[k + i*qDim + j*m*qDim];  
                    //macroscopic velocity x 
                    u[i*dDim + j*m*dDim] = u[i*dDim + j*m*dDim] 
                                              + e[k] * fIn[k + i*qDim + j*m*qDim]; 
                    //macroscopic velocity y
                    u[1 + i*dDim + j*m*dDim] = u[1 + i*dDim + j*m*dDim] 
                                                  + e[k + qDim] * fIn[k + i*qDim + j*m*qDim];

                }
                u[i*dDim + j*m*dDim]     = u[i*dDim + j*m*dDim] / rho[i + j*m];
                u[1 + i*dDim + j*m*dDim] = u[1 + i*dDim + j*m*dDim] / rho[i + j*m];
            }
        }
    }
    
     
    // Collision step
      #pragma omp target teams distribute parallel for private(i,j,k,cu,fIn_s,fIn_a,fEq_s,fEq_a,omega_trt) collapse(2)
    for (i = 0; i < m; ++i)
    {
        for (j = 0; j < n; ++j)
            {
                for (k = 0; k < qDim; ++k)
                {
                    cu = 3.f*(static_cast<float>(e[k]) * u[i*dDim + j*m*dDim] 
                         + static_cast<float>(e[k + qDim]) * u[1 + i*dDim + j*m*dDim] );

                    fEq[k + i*qDim + j*m*qDim] = rho[i + j*m] * t[k] * (1. + cu + 1./2.*(cu * cu)
                                 - 3./2. * (pow(u[i*dDim + j*m*dDim],2) + pow(u[1 + i*dDim + j*m*dDim],2)) );
                    fOut[k + i*qDim + j*m*qDim] = fIn[k + i*qDim + j*m*qDim] 
                                                  - omega * ( fIn[k + i*qDim + j*m*qDim]  - fEq[k + i*qDim + j*m*qDim] );

                    /*
                    // TRT
                    fIn_s = (fIn[k + i*qDim + j*m*qDim] + fIn[opp[k] + i*qDim + j*m*qDim]) * 0.5f;
                    fIn_a = (fIn[k + i*qDim + j*m*qDim] - fIn[opp[k] + i*qDim + j*m*qDim]) * 0.5f;
                    fEq_s = (fEq[k + i*qDim + j*m*qDim] + fEq[opp[k] + i*qDim + j*m*qDim]) * 0.5f;
                    fEq_a = (fEq[k + i*qDim + j*m*qDim] - fEq[opp[k] + i*qDim + j*m*qDim]) * 0.5f;

                    omega_trt = - (fIn_s - fEq_s) / tau_s - (fIn_a - fEq_a) / tau_a;
                    //fOut(i,j,k) = fIn(i,j,k) + omega_trt;
                    fOut[k + i*qDim + j*m*qDim] = fIn[k + i*qDim + j*m*qDim] + omega_trt;
                    */
                    
                }
            }
    }
      
    // Obstacle (bounce back)
    #pragma omp target teams distribute parallel for private(i,j,k) collapse(2) device(0) 
    for (i = 0; i < m; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            if (isSolid[i + j*m] == 1)
            {
	        for (k = 0; k < qDim; ++k)
                {                    
                    fOut[k + i*qDim + j*m*qDim] = fIn[opp[k] + i*qDim + j*m*qDim];
                }
            }
        }
    }
        
    // Streaming
    #pragma omp target teams distribute parallel for private(i,j,k,i_p,j_p) collapse(2)
    for (i = 0; i < m; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            for (k = 0; k < qDim; ++k)
            {
        
                i_p = i + e[k];
                j_p = j + e[k + qDim]; 
            
                if (i_p < 0)
                     { i_p = m-1; }
                else if (i_p > (m-1))
                     { i_p = 0; }
                  
                if (j_p < 0)
                    { j_p = n-1; }
                else if (j_p > (n-1))
                    { j_p = 0; }
                
               fIn[k + i_p*qDim + j_p*m*qDim] = fOut[k + i*qDim + j*m*qDim];
                
           }
                         
        }
    }
    
    
    //Solution output
    if (outputFlag == 1 && simTime % timePlot == 0)
    {
        #pragma omp target update from(u)
    ofstream fout;
    std::ostringstream fileNameStream("u_x");
    fileNameStream << "u_x_" << simTime << ".csv";
    std::string fileName = fileNameStream.str();
    fout.open (fileName.c_str());
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++) 
        {
            if (i == m - 1)
               fout << u[i*dDim + j*m*dDim] << std::endl;
               fout << u[i*dDim + j*m*dDim] << ",";
        }
    }
    fout.close();

    ofstream fout2;
    std::ostringstream fileNameStream2("u_y");
    fileNameStream2 << "u_y_" << simTime << ".csv";
    std::string fileName2 = fileNameStream2.str();
    fout2.open (fileName2.c_str());
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++) 
        {
            if (i == m - 1)
               fout2 << u[1 + i*dDim + j*m*dDim] << std::endl;
               fout2 << u[1 + i*dDim + j*m*dDim] << ",";
        }
    }
    fout2.close();
    }

    }  //end time loop

    
    double wclock_time =  omp_get_wtime() - start; 
    printf("Wall clock time = %lf s.\n", wclock_time);

    return 0;
}


