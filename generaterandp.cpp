#include "traj.h"
void generate_rand_sphere(float a0, float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd],
                          float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd],
                          int q[2][n_partd], int m[2][n_partd], int nt[2], float dt[2])
{
    // spherical plasma set plasma parameters
    float Temp[2] = {Temp_e, Temp_d}; // in K convert to eV divide by 1.160451812e4
    // initial bulk electron, ion velocity
    float v0[2][3] = {{0, 0, 0 /*1e6*/}, {0, 0, 0}};

    float r0 = 2 * a0; // if sphere this is the radius
    float area = 4 * pi * r0 * r0;
    float volume = 4 / 3 * pi * r0 * r0 * r0;

    // calculated plasma parameters
    float Density_e = n_partd / volume * r_part_spart;
    cout << "electron Density = " << Density_e << "/m^3 \n";
    float plasma_freq = sqrt(Density_e * e_charge * e_charge_mass / (mp[0] * epsilon0)) / (2 * pi);
    float plasma_period = 1 / plasma_freq;
    float Debye_Length = sqrt(epsilon0 * kb * Temp[0] / (Density_e * e_charge * e_charge));
    float vel_e = sqrt(kb * Temp[0] / (mp[0] * e_mass));
    float Tv = a0 / vel_e; // time for electron to move across 1 cell
    float Tcyclotron = 2.0 * pi * mp[0] / (e_charge_mass * Bmax0);
    float TDebye = Debye_Length / vel_e;
    float TE = sqrt(2 * a0 / e_charge_mass / Emax0);
    // set time step to allow electrons to gyrate if there is B field or to allow electrons to move slowly throughout the plasma distance

    dt[0] = 4 * min(min(min(TDebye, min(Tv / md_me, Tcyclotron) / 4), plasma_period / ncalc0[0] / 4), TE / ncalc0[0]) / 2; // electron should not move more than 1 cell after ncalc*dt and should not make more than 1/4 gyration and must calculate E before the next 1/4 plasma period
    dt[1] = dt[0] * md_me;
    //  float mu0_4pidt[2]= {mu0_4pi/dt[0],mu0_4pi/dt[1]};
    cout << "v0 electron = " << v0[0][0] << "," << v0[0][1] << "," << v0[0][2] << endl;

    // set initial positions and velocity
    float sigma[2] = {sqrt(kb * Temp[0] / (mp[0] * e_mass)), sqrt(kb * Temp[1] / (mp[1] * e_mass))};
    long seed;
    gsl_rng *rng;                        // random number generator
    rng = gsl_rng_alloc(gsl_rng_rand48); // pick random number generator
    time_t myTime;
    seed = time(&myTime);
    cout << "seed=" << seed << "\n";
    gsl_rng_set(rng, seed); // set seed

    for (int p = 0; p < 2; p++)
    {
        int na = 0;
        for (int n = 0; n < nback; ++n) // set number of particles per cell in background
        {
            for (int k = 1; k < n_space_divz - 1; ++k)
            {
                for (int j = 1; j < n_space_divy - 1; ++j)
                    for (int i = 1; i < n_space_divx - 1; ++i)
                    {
                        pos0x[p][na] = ((float)(i - n_space_divx / 2) + (float)rand() / RAND_MAX) * a0;
                        pos0y[p][na] = ((float)(j - n_space_divy / 2) + (float)rand() / RAND_MAX) * a0;
                        pos0z[p][na] = ((float)(k - n_space_divz / 2) + (float)rand() / RAND_MAX) * a0;
                        pos1x[p][na] = pos0x[p][na];
                        pos1y[p][na] = pos0y[p][na];
                        pos1z[p][na] = pos0z[p][na];
                        q[p][na] = qs[p];
                        m[p][na] = mp[p];
                        na++;
                    }
                //         cout << pos1z[p][na - 1] << " ";
            }
        }

#pragma omp parallel for
        for (int n = na; n < n_partd; n++)
        {
            float r = r0 * pow(gsl_ran_flat(rng, 0, 1), 0.3333333333);
            double x, y, z;
            gsl_ran_dir_3d(rng, &x, &y, &z);
            pos0x[p][n] = r * x;
            pos1x[p][n] = pos0x[p][n] + (gsl_ran_gaussian(rng, sigma[p]) + v0[p][0]) * dt[p];
            pos0y[p][n] = r * y;
            pos1y[p][n] = pos0y[p][n] + (gsl_ran_gaussian(rng, sigma[p]) + v0[p][1]) * dt[p];
            pos0z[p][n] = r * z;
            pos1z[p][n] = pos0z[p][n] + (gsl_ran_gaussian(rng, sigma[p]) + v0[p][2]) * dt[p];
            q[p][n] = qs[p];
            m[p][n] = mp[p];
            //         nt[p] += q[p][n];
        }
    }
#pragma barrier
    gsl_rng_free(rng); // dealloc the rng
}
void generate_rand_cylinder(float a0, float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd],
                            float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd],
                            int q[2][n_partd], int m[2][n_partd], int nt[2], float dt[2])
{
    // set plasma parameters
    // float mp[2]= {9.10938356e-31,3.3435837724e-27}; //kg
    //  int qs[2] = {-1, 1}; // Sign of charge
    // spherical plasma radius is 1/8 of total extent.
    float Temp[2] = {Temp_e, Temp_d}; // in K convert to eV divide by 1.160451812e4
    // initial bulk electron, ion velocity
    float v0[2][3] = {{0, 0, -1e7 /*1e6*/}, {0, 0, 0}};

    float r0 = a0; // the radius
    float area = pi * r0 * r0;
    float volume = pi * r0 * r0 * n_space * a0;
    // float volume=((posHp[0]-posLp[0])*(posHp[1]-posLp[1])*(posHp[2]-posLp[2]))); //cube

    // calculated plasma parameters
    cout << "initial e Temperature, = " << Temp_e / 11600 << "eV, initial d Temperature, = " << Temp_d / 11600 << " eV\n";
    float Density_e = n_partd / volume * r_part_spart;
    cout << "initial density = " << Density_e << endl;
    float initial_current = Density_e * e_charge * v0[0][2] * area;
    cout << "initial current = " << initial_current << endl;
    float Bmaxi = initial_current * 2e-7 / r0;
    cout << "initial Bmax = " << Bmaxi << endl;
    float plasma_freq = sqrt(Density_e * e_charge * e_charge_mass / (mp[0] * epsilon0)) / (2 * pi);
    float plasma_period = 1 / plasma_freq;
    float Debye_Length = sqrt(epsilon0 * kb * Temp[0] / (Density_e * e_charge * e_charge));
    float vel_e = sqrt(kb * Temp[0] / (mp[0] * e_mass));
    float Tv = a0 / vel_e; // time for electron to move across 1 cell
    float Tcyclotron = 2.0 * pi * mp[0] / (e_charge_mass * Bmax0);
    float TDebye = Debye_Length / vel_e;
    float TE = sqrt(2 * a0 / e_charge_mass / Emax0);
    // set time step to allow electrons to gyrate if there is B field or to allow electrons to move slowly throughout the plasma distance

    dt[0] = 4 * min(min(min(TDebye, min(Tv / md_me, Tcyclotron) / 4), plasma_period / ncalc0[0] / 4), TE / ncalc0[0]) / 2; // electron should not move more than 1 cell after ncalc*dt and should not make more than 1/4 gyration and must calculate E before the next 1/4 plasma period
    dt[1] = dt[0] * md_me;
    //  float mu0_4pidt[2]= {mu0_4pi/dt[0],mu0_4pi/dt[1]};
    cout << "v0 electron = " << v0[0][0] << "," << v0[0][1] << "," << v0[0][2] << endl;

    // set initial positions and velocity
    float sigma[2] = {sqrt(kb * Temp[0] / (mp[0] * e_mass)), sqrt(kb * Temp[1] / (mp[1] * e_mass))};
    long seed;
    gsl_rng *rng;                        // random number generator
    rng = gsl_rng_alloc(gsl_rng_rand48); // pick random number generator
    seed = 1670208073;                   // time(NULL);
    cout << "seed=" << seed << "\n";
    gsl_rng_set(rng, seed); // set seed

    for (int p = 0; p < 2; p++)
    {
        int na = 0;
        for (int k = 1; k < n_space_divz - 1; ++k)
            for (int j = 1; j < n_space_divy - 1; ++j)
                for (int i = 1; i < n_space_divx - 1; ++i)
                {
                    pos0x[p][na] = (float)(i - n_space_divx / 2) * a0;
                    pos0y[p][na] = (float)(j - n_space_divy / 2) * a0;
                    pos0z[p][na] = (float)(k - n_space_divz / 2) * a0;
                    pos1x[p][na] = pos0x[p][na];
                    pos1y[p][na] = pos0y[p][na];
                    pos1z[p][na] = pos0z[p][na];
                    q[p][na] = qs[p];
                    m[p][na] = mp[p];
                    na++;
                }
        for (int n = na; n < n_partd; n++)
        {
            float r = r0 * pow(gsl_ran_flat(rng, 0, 1), 0.5);
            double x, y, z;
            z = gsl_ran_flat(rng, -1.0, 1.0) * a0 * n_space / 2;
            gsl_ran_dir_2d(rng, &x, &y);
            pos0x[p][n] = r * x;
            pos1x[p][n] = pos0x[p][n] + (gsl_ran_gaussian(rng, sigma[p]) + v0[p][0]) * dt[p];
            pos0y[p][n] = r * y;
            pos1y[p][n] = pos0y[p][n] + (gsl_ran_gaussian(rng, sigma[p]) + v0[p][1]) * dt[p];
            pos0z[p][n] = z;
            pos1z[p][n] = pos0z[p][n] + (gsl_ran_gaussian(rng, sigma[p]) + v0[p][2]) * dt[p];
            //          if (n==0) cout << "p = " <<p <<", sigma = " <<sigma[p]<<", temp = " << Temp[p] << ",mass of particle = " << mp[p] << dt[p]<<endl;
            q[p][n] = qs[p];
            m[p][n] = mp[p];
            nt[p] += q[p][n];
        }
    }
    gsl_rng_free(rng); // dealloc the rng
}