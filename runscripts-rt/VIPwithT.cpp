//
//  VIPwithT.cpp
//  derived from VIPcells2.cpp
//  want to add T-current (and M-current) to VIP with D-current
//  T-current from TCcell.cpp
//
//
//  Created by McCarthy, Michelle M on 4/11/23.
//

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream> //this is the library needed for file streaming
#include <stdlib.h>
#include <time.h> // needed to seed srand

/*-- constants --*/

#define  SVCOUNT     11 // number of state variables
#define  SVCOUNT2    2  // number of state variables
#define  VIPCOUNT    2  // number of VIP cells
#define  START        0
#define  END        5000 //ms
#define  TIMESTEP   0.05
#define  Fbeg       0.0001 // 0.0001, beginning frequency is 0.001 if 1 Hz Hz, i.e. 1/1000
#define  Fend      0.05 //0.05, end frequency in Hz is 0.1 if 100 Hz, i.e. 100/1000

/*-- function prototypes --*/

float box_muller(float m, float s);    /* normal random variate generator */
int IntRK4( int svCount, int svCount2, double sv[SVCOUNT][SVCOUNT2], double time, double dt, double LFP[3], double Ifreq);
void Derive( int svCount, int svCount2, double y[SVCOUNT][SVCOUNT2], double k[SVCOUNT][SVCOUNT2], double time, double LFP[3], double Ifreq);
double rnd();

using namespace std;

/*-- main program --*/


int main( void )

{
    
    /*** Output to File ****/
    ofstream outputFile("results.txt", ios::out);
    ofstream outputFile2("results2.txt", ios::out);
    ofstream outputFile3("results3.txt", ios::out);
    ofstream outputFile4("results4.txt", ios::out);
    ofstream outputFile5("results5.txt", ios::out); // LFP for VIP cells
    
    if (!outputFile)
    {
        cerr << "File could not be opened" << endl;
        exit(1);
    }
    
    /* Seed random number generator */
    srand ( time(NULL) );
    
    /*** define variables ****/
    double sv[ SVCOUNT ][SVCOUNT2]; double LFP[3];
    double svp[VIPCOUNT]; double ispk[VIPCOUNT]; double ispktime[VIPCOUNT];
    double time; int itime; int itime2; int ntime;
    
    int z;
    
    double ZAPfreq = Fend*1000;
    double Izap0 = 0;
    double Izap0_pre = 0;
    double ptime = 0;
    double SineWave = 0;
    
    /*** set initial conditions for VIP cells ****/
    for (z=0; z<VIPCOUNT; z++)
    {
        sv[0][z] = -63; //+ 5*sqrt(TIMESTEP)*box_muller(0,1);
        sv[1][z] = 1/(1+exp((sv[0][z]+58.3)/6.7));
        sv[2][z] = 0.128*exp(-(sv[0][z]+50)/18) / (0.128*exp(-(sv[0][z]+50)/18)  +  4/(1+exp(-(sv[0][z]+27)/5)));
        sv[3][z] = 1/(1+ exp(-(sv[0][z]+12.4)/6.8));
        sv[4][z] = 3.209*0.0001*(sv[0][z]+30)/(1-exp(-(sv[0][z]+30)/9)) / (3.209*0.0001*(sv[0][z]+30)/(1-exp(-(sv[0][z]+30)/9))   -  3.209*0.0001*(sv[0][z]+30)/(1-exp((sv[0][z]+30)/9)));
        sv[8][z] = 1/(1 + exp(-(sv[0][z]+50)/20));
        sv[9][z] = 1/(1 + exp((sv[0][z]+70)/6));
        
        sv[5][z] = 0.0001;
        
        sv[ 6 ][z] = 1 / ( 1 + exp(-(sv[0][z]+57)/6.2) );
        sv[ 7 ][z] = 1 / ( 1+exp((sv[0][z]+81)/4) );
        
        sv[ 10 ][z] = 0; // sinusoidal input
        
        svp[z] = 0; ispk[z] = 0; ispktime[z] = 0;
    }
    double C = 0;
    double Ifreq = 5; // start freq
    double dIfreq = 5; // step size
    while (Ifreq <= 80){ // end freq
        
    time = START + C*END;
        
    
    /* start main time loop - loop until stopping time reached */
    while ( time <  END + C*END )
    {
        ntime = time; // makes time an integer
        itime = ntime % 5;  // itime is an integer mod 5 --> (fs = 200), mod2 --> fs = 500
        
        if (time >= 0)
            Izap0 = 1*sin(2*3.14159*Fbeg*(END-time)*pow((Fend/Fbeg),((END-time)/END)));

        SineWave = 0.02*sin(2*M_PI/(1000/Ifreq)*time);
        
        IntRK4(  SVCOUNT, SVCOUNT2, sv, time, TIMESTEP, LFP, Ifreq);
        
        outputFile<<time<<","<<sv[0][0]<<","<<Izap0<<","<<ZAPfreq<<","<<SineWave<<","<<Ifreq<<endl;
        
        
    //********** Find frequency of Zap current ********//
        if (Izap0 <= 0 && Izap0_pre > 0)
        {
            ZAPfreq = 1000/(time - ptime);
            ptime = time;
        }
        
        /********** Raster Plot for VIPs **********/
        for (z=0; z<VIPCOUNT; z++)
        {
            if (time >= 500)
            {
                if (sv[0][z] > 0 && svp[z] <= 0)
                {
                    ispk[z] = z;
                    ispktime[z] = time;
                    outputFile3<<ispk[z]<<","<<ispktime[z]<<endl;
                }
                else
                    ispk[z] = 0;
            }
        }
        
        /********** Filtered LFP Output for VIPs **********/
        if(time >= 200)
        {
            if ( itime == 0 && itime2 != 0)
            {
                outputFile5<<time<<","<<LFP[0]<<endl;
            }
        }
        
        
        for (z=0; z<VIPCOUNT; z++)
            svp[z] = sv[0][z];
        
        Izap0_pre = Izap0;
        itime2 = itime;
        
        time += TIMESTEP;
    }
        Ifreq = Ifreq + dIfreq;
        C = C+1;
    }
    
    return 0;
}


/*---------------------- Integration Methods ----------------------
 
 *-- IntRK4() -------------------------------------------------------
 *
 *--- general purpose Runge-Kutta integrator (fourth order)
 *    code is modified from "Numerical Recipes"
 *
 *--INPUTS
 *   int     svCount    = state variable count
 *   float  *stateVar   = state variable value array
 *   double *derivative = derivative value array
 *   float  *time      = time variable address
 *   float   dt         = timestep
 *-------------------------------------------------------------------
 */

int IntRK4( int svCount, int svCount2, double sv[SVCOUNT][SVCOUNT2], double time, double dt, double LFP[3], double Ifreq)
{
    int  i;                      /* loop counter */
    int  j;/* loop counter */
    
    /*-- temporary arrays --*/
    static double y [ SVCOUNT ][ SVCOUNT2 ];
    static double k1[ SVCOUNT ][ SVCOUNT2 ];
    static double k2[ SVCOUNT ][ SVCOUNT2 ];
    static double k3[ SVCOUNT ][ SVCOUNT2 ];
    static double k4[ SVCOUNT ][ SVCOUNT2 ];
    
    for ( i=0; i < SVCOUNT; i++ )       /* remember initial values */
    {
        for ( j=0; j < SVCOUNT2; j++)
        {
            y[ i ][j] = sv[ i ][j];
        }
    }
    
    /*-- pass 1 --*/
    Derive( SVCOUNT, SVCOUNT2, y, k1, time, LFP, Ifreq);     /* get derivatives */
    
    for ( i=0; i < SVCOUNT; i++ )       /* update state variables */
    {
        for ( j=0; j < SVCOUNT2; j++)
        {
            y[ i ][j] = sv[ i ][j] + k1[ i ][j] * dt/2;
        }
    }
    
    /*-- pass 2 --*/
    time += dt/2;
    Derive( SVCOUNT, SVCOUNT2, y, k2, time, LFP, Ifreq);     /* get derivatives */
    
    for ( i=0; i < SVCOUNT; i++ )       /* update state variables */
    {
        for ( j=0; j < SVCOUNT2; j++)
        {
            y[ i ][j] = sv[ i ][j] + k2[ i ][j] * dt/2;
        }
    }
    
    /*- pass 3 --*/
    Derive( SVCOUNT, SVCOUNT2, y, k3, time, LFP, Ifreq);     /* get derivatives */
    
    for ( i=0; i < SVCOUNT; i++ )
    {
        for ( j=0; j < SVCOUNT2; j++)
        {
            y[ i ][j] = sv[ i ][j] + k3[ i ][j] * dt;    /* update state variables */
        }
    }
    
    /*-- pass 4 --*/
    time += dt/2;
    Derive( SVCOUNT, SVCOUNT2, y, k4, time, LFP, Ifreq);  /* get derivatives */
    for ( i=0; i < SVCOUNT; i++ )
    {
        for ( j=0; j < SVCOUNT2; j++)
        {
            sv[ i ][j] = sv[ i ][j]  + ( k1[ i ][j] + 2*k2[ i ][j] + 2*k3[ i ][j] +k4[ i ][j] ) * dt/6;
        }
    }
    
    return 1;
}


void Derive( int svCount, int svCount2, double y[SVCOUNT][SVCOUNT2], double k[SVCOUNT][SVCOUNT2], double time, double LFP[3], double Ifreq)
{
    int z;
    
    double gna = 112.5; // 112.5 conductances in units of mS/cm^2
    double gk = 225; // 225
    double gm = 1;
    double gd = 3; //1 mS/cm2
    double gl = 0.25; //0.1
    double Ena = 50; // reversal potentials in units of mV
    double Ek  = -90; // -90
    double El  = -70; // -67
    double Ei = -80;  //-80
    double Ee = 0;
    double taui = 5; // time constant of decay of GABAa current in units of ms
    double Cm = 1;
    double SineWave;
    
    // Calcium parameters
    double F = 96489; // Faraday's constant in Coulombs/mole
    //    double F = 96; // Faraday's constant in Coulombs/mM
    double R = 8.31; // Gas constant in Joules/(mole*T_Kelvin)
    //    double R = 0.00831;
    double Temp = 310;  // Temperature in Kelvin, 37 degrees Celsius
    //    double Cao = 0.002; // M
    //    double Cainf = 0.00000024;  // M (240nM)
    double Cao = 0.003;
    double Cainf = 0.00000001;
    double tauCa = 5;
    double VIParea = 0.00029;
    double Pca   = 40;
    
    /*** Applied current provides heterogeneity to the system ****/
    double Iapp[VIPCOUNT];
    Iapp[0] = 4.5;
    for (z=1; z<VIPCOUNT; z++)
        Iapp[z] = Iapp[z-1] + 0.00;
    
    
    double SumVIP = 0; // sum of all VIP voltages
    for (z=0; z<VIPCOUNT; z++)
        SumVIP = SumVIP + y[0][z];
    
    double Sum5 = 0; // sum of all VIP inhibition variables
    for (z=0; z<VIPCOUNT; z++)
        Sum5 = Sum5 + y[5][z];
    
    /********* ZAP CURRENT ********/
    double Izap = 0;
    if (time >= 0)
             Izap = 0.002*sin(2*3.14159*Fbeg*(END-time)*pow((Fend/Fbeg),((END-time)/END)));
    
    //**** Sinusoidal input *****//
 //    k[ 10 ][0] = 0.02*sin(2*M_PI/(1000/Ifreq)*time); //0.1 is ~16 Hz, 0.25 is 40 Hz
    k[ 10 ][0] = 0.02*sin(0.1*time); //0.1 is ~16 Hz, 0.25 is 40 Hz
    
    SineWave = 0.02*sin(2*M_PI/(1000/Ifreq)*time);
    
    
    /*** VIP equations ****/
    double ainfD[VIPCOUNT]; double binfD[VIPCOUNT]; double minf[VIPCOUNT]; double hinf[VIPCOUNT]; double tauh[VIPCOUNT]; double ninf[VIPCOUNT]; double taun[VIPCOUNT]; double mtinf[VIPCOUNT]; double htinf[VIPCOUNT]; double taumt[VIPCOUNT]; double tauht[VIPCOUNT]; double Gt[VIPCOUNT];
    LFP[0] = 0;
    for (z=0; z<VIPCOUNT; z++)
    {
        // Na-current
        minf[z] = 1/(1+exp(-(y[0][z]+24)/11.5));
        hinf[z] = 1/(1+exp((y[0][z]+58.3)/6.7));
        tauh[z] =  0.5 + 14/(1+exp((y[0][z]+60)/12));
        
        // Fast K-current
        ninf[z] = 1/(1+ exp(-(y[0][z]+12.4)/6.8));
        taun[z] = (0.087 + 11.4/(1+exp((y[0][z]+14.6)/8.6))) * (0.087 + 11.4/(1+exp(-(y[0][z]-1.3)/18.7)));
        
        // Na and K and M gating equations
        k[ 2 ][z] = (hinf[z] - y[2][z])/tauh[z]; // derivative equation for h, Goloumb version
        k[ 3 ][z] = (ninf[z] - y[3][z])/taun[z];
        k[ 4 ][z] = (3.209*0.0001*(y[0][z]+30)/(1-exp(-(y[0][z]+30)/9)))*(1-y[4][z]) + 3.209*0.0001*(y[0][z]+30)/(1-exp((y[0][z]+30)/9))*y[4][z];  // derivative equation for w (for m-current) -- Mainen and Sejowski
        
        // T-current
        mtinf[z] = 1 / ( 1 + exp(-(y[0][z]+57)/6.2) );
        htinf[z] = 1 / ( 1+exp((y[0][z]+81)/4) );
        taumt[z] = 0.612 + 1 / ( exp((y[0][z]+16.8)/18.2) + exp(-(y[0][z]+132)/16.7) );
        if (y[0][z] < -81.0)
            tauht[z] = 1 * exp((y[0][z]+467)/66.6);
        else
            tauht[z] = 28 + exp(-(y[0][z]+22)/10.5);
        
        // Using V (instead of mV) and thus multiplying by 0.001 in formula for Gt to get units correct.  Unit of G is Coulombs.
        // Or alternatively can recognize that RT/F = 26.7 mV and write Gt equation this way.
        Gt[z] = (4*F*F*0.001*y[0][z]/(R*Temp))*(Cainf - Cao*exp(-2*F*0.001*y[0][z]/(R*Temp))) / (1.0 - exp(-2*F*0.001*y[0][z]/(R*Temp)));
        
        
        // T-current gating equations
        k[ 6 ][z] = (mtinf[z]-y[6][z])/taumt[z];  // derivative equation for mt (for t-current, low threshold inactivating Ca++ current)
        k[ 7 ][z] = (htinf[z]-y[7][z])/tauht[z];  // derivative equation for ht (for t-current, low threshold inactivating Ca++ current)
        
        // D-current
        ainfD[z] =  1/(1 + exp(-(y[0][z]+50)/20));
        binfD[z] =  1/(1 + exp((y[0][z]+70)/6));
        
        // D-current gating equations
        k[8][z] = (ainfD[z] - y[8][z])/2;
        k[9][z] = (binfD[z] - y[9][z])/150; // 150
        
            Iapp[z] = 9.8; // 9 for rhythmic / sinusoidal input
        
        Pca   = 80;
        gm = 1;

        // Voltage equation for VIP cell
        k[ 0 ][z] = (1/Cm)*(-gna*y[2][z]*(minf[z]*minf[z]*minf[z])*(y[0][z]-Ena)
                            -gk*(y[3][z]*y[3][z])*(y[0][z]-Ek)
                            -gd*y[8][z]*y[8][z]*y[8][z]*y[9][z]*(y[0][z]-Ek)
                            -gm*y[4][z]*(y[0][z]-Ek)
                            -Pca*VIParea*(y[6][z]*y[6][z])*y[7][z]*Gt[z]
                            -gl*(y[0][z]-El)
                            )
     //   +Izap
  //      + SineWave // sinusoidal input
        //    +20*sqrt(0.05)*box_muller(0,1)
             + Iapp[z];
        
        LFP[0] = y[0][z] + LFP[0];
        
        /*** Synaptic equations ****/
        double sgi[VIPCOUNT];
        taui = 10;
        sgi[z] = 2*(1+tanh(y[0][z]/4)); // i to i
        k[ 5 ][z] = sgi[z]*(1-y[5][z])-y[5][z]/taui;  // derivative equation for inhibitory GABAa synaptic activity (i to i kinetics)
        
    }
    
    
}


double rnd()  {   // returns a uniform random number between 0 and 1
    return((double)rand()/RAND_MAX);
}


double randgauss( double min, double max, double sigma, double centre)
{
    double gauss = (min + (max-min) * (double)rand()/RAND_MAX); //create random domain between [min,max]
    
    //    double tmp = (random-centre)/sigma;
    //    double gauss= exp(-tmp*tmp/2); //gaussian formula
    
    return gauss;
}



float box_muller(float m, float s)    /* normal random variate generator */  //generates Gaussian random numbers
{                        /* mean m, standard deviation s */
    float x1, x2, w, y1;
    static float y2;
    static int use_last = 0;
    
    if (use_last)                /* use value from previous call */
    {
        y1 = y2;
        use_last = 0;
    }
    else
    {
        do {
            x1 = 2.0 * randgauss(0,1,1,0) - 1.0;
            x2 = 2.0 * randgauss(0,1,1,0) - 1.0;
            w = x1 * x1 + x2 * x2;
        } while ( w >= 1.0 );
        
        w = sqrt( (-2.0 * log( w ) ) / w );
        y1 = x1 * w;
        y2 = x2 * w;
        use_last = 1;
    }
    
    return( m + y1 * s );
}












