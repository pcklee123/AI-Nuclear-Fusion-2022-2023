/* TS3.cpp
This contains the main loop for the program. Most of the initialization occurs here, and time steps are iterated through.
For settings (as to what to calculate, eg. E / B field, E / B force) go to the defines in include/traj.h
*/
#include "include/traj.h"
// sphere
// 0,number of "super" electrons, electron +deuteriom ions, total
unsigned int n_space_div[3] = {n_space_divx, n_space_divy, n_space_divz};
unsigned int n_space_div2[3] = {n_space_divx2, n_space_divy2, n_space_divz2};
par par1;
par *par = &par1;
particles particl1;
particles *pt = &particl1; //= alloc_particles( par);
// string outpath;
ofstream info_file;
int main()
{
    timer.mark(); // Yes, 3 time marks. The first is for the overall program dt
    timer.mark(); // The second is for compute_d_time
    timer.mark(); // The third is for start up dt
    double t = 0;

    const unsigned int n_cells = n_space_divx * n_space_divy * n_space_divz;

    // position of particle and velocity: stored as 2 positions at slightly different times [3 components][2 types of particles][number of particles]
    /** CL: Ensure that pos0/1.. contain multiple of 64 bytes, ie. multiple of 16 floats **/
    //*
    auto *pos = reinterpret_cast<float(&)[2][3][2][n_partd]>(*((float *)_aligned_malloc(sizeof(float) * par->n_part[0] * 2 * 3 * 2, par->cl_align)));

    auto *pos0x = reinterpret_cast<float(&)[2][n_partd]>(*(float *)(pos[0][0]));
    auto *pos0y = reinterpret_cast<float(&)[2][n_partd]>(*(float *)(pos[0][1]));
    auto *pos0z = reinterpret_cast<float(&)[2][n_partd]>(*(float *)(pos[0][2]));
    auto *pos1x = reinterpret_cast<float(&)[2][n_partd]>(*(float *)(pos[1][0]));
    auto *pos1y = reinterpret_cast<float(&)[2][n_partd]>(*(float *)(pos[1][1]));
    auto *pos1z = reinterpret_cast<float(&)[2][n_partd]>(*(float *)(pos[1][2]));
    pt->pos = pos;
    pt->pos0 = reinterpret_cast<float(*)>(pos[0]);
    pt->pos1 = reinterpret_cast<float(*)>(pos[1]);
    pt->pos0x = pos0x;
    pt->pos0y = pos0y;
    pt->pos0z = pos0z;
    pt->pos1x = pos1x;
    pt->pos1y = pos1y;
    pt->pos1z = pos1z;
    // */
    /*
        auto *pos0x = reinterpret_cast<float(&)[2][n_partd]>(*((float *)_aligned_malloc(sizeof(float) * n_partd * 2, par->cl_align))); // new float[2][n_partd];
        auto *pos0y = reinterpret_cast<float(&)[2][n_partd]>(*((float *)_aligned_malloc(sizeof(float) * n_partd * 2, par->cl_align))); // new float[2][n_partd];
        auto *pos0z = reinterpret_cast<float(&)[2][n_partd]>(*((float *)_aligned_malloc(sizeof(float) * n_partd * 2, par->cl_align))); // new float[2][n_partd];
        auto *pos1x = reinterpret_cast<float(&)[2][n_partd]>(*((float *)_aligned_malloc(sizeof(float) * n_partd * 2, par->cl_align))); // new float[2][n_partd];
        auto *pos1y = reinterpret_cast<float(&)[2][n_partd]>(*((float *)_aligned_malloc(sizeof(float) * n_partd * 2, par->cl_align))); // new float[2][n_partd];
        auto *pos1z = reinterpret_cast<float(&)[2][n_partd]>(*((float *)_aligned_malloc(sizeof(float) * n_partd * 2, par->cl_align))); // new float[2][n_partd];
                                                                                                                                        */

    //    charge of particles
    auto *q = static_cast<int(*)[n_partd]>(_aligned_malloc(2 * n_partd * sizeof(int), alignment)); // charge of each particle +1 for H,D or T or -1 for electron can also be +2 for He for example
    auto *m = static_cast<int(*)[n_partd]>(_aligned_malloc(2 * n_partd * sizeof(int), alignment)); // mass of of each particle not really useful unless we want to simulate many different types of particles
    pt->q = q;
    pt->m = m;
    // reduced particle position dataset for printing/plotting
    // auto *posp = new float[2][n_output_part][3];
    // auto *KE = new float[2][n_output_part];

    /** CL: Ensure that Ea/Ba contain multiple of 64 bytes, ie. multiple of 16 floats **/
    auto *E = reinterpret_cast<float(&)[3][n_space_divz][n_space_divy][n_space_divx]>(*fftwf_alloc_real(3 * n_cells));                             // selfgenerated E field
    auto *Ee = new float[3][n_space_divz][n_space_divy][n_space_divx];                                                                             // External E field
    auto *Ea = static_cast<float(*)[n_space_divy][n_space_divx][3][ncoeff]>(_aligned_malloc(sizeof(float) * n_cells * 3 * ncoeff, par->cl_align)); // coefficients for Trilinear interpolation Electric field

    auto *B = reinterpret_cast<float(&)[3][n_space_divz][n_space_divy][n_space_divx]>(*fftwf_alloc_real(3 * n_cells)); // new float[3][n_space_divz][n_space_divy][n_space_divx];
    auto *Be = new float[3][n_space_divz][n_space_divy][n_space_divx];
    auto *Ba = static_cast<float(*)[n_space_divy][n_space_divx][3][ncoeff]>(_aligned_malloc(sizeof(float) * n_cells * 3 * ncoeff, par->cl_align)); // coefficients for Trilinear interpolation Magnetic field

    auto *V = reinterpret_cast<float(&)[1][n_space_divz][n_space_divy][n_space_divx]>(*fftwf_alloc_real(n_cells));

    auto *np = static_cast<float(*)[n_space_divz][n_space_divy][n_space_divx]>(_aligned_malloc(2 * n_cells * sizeof(float), alignment));
    auto *npt = static_cast<float(*)[n_space_divy][n_space_divx]>(_aligned_malloc(n_cells * sizeof(float), alignment));
    auto *currentj = static_cast<float(*)[3][n_space_divz][n_space_divy][n_space_divx]>(_aligned_malloc(2 * 3 * n_cells * sizeof(float), alignment));
    auto *jc = static_cast<float(*)[n_space_divz][n_space_divy][n_space_divx]>(_aligned_malloc(3 * n_cells * sizeof(float), alignment));

    log_headers();

    cout << std::scientific;
    cout.precision(1);
    cerr << std::scientific;
    cerr.precision(3);

    int total_ncalc[2] = {0, 0}; // particle 0 - electron, particle 1 deuteron
    cout << "Start up time = " << timer.replace() << "s\n";
    // startup stuff set output path opencl and print initial info
    info(par); // printout initial info.csv file
#define generateRandom
#ifdef generateRandom
#ifdef sphere
    generate_rand_sphere(pt, par);
#endif // sphere
#ifdef cylinder
    generate_rand_cylinder(pt, par);
#endif // cylinder
#else
    generateParticles(a0, r0, qs, mp, pt, nt);
#endif

    // get limits and spacing of Field cells
    generateField(Ee, Be);

    cout << "Set initial random positions: " << timer.replace() << "s\n";

    fftwf_init_threads();

    int i_time = 0;
    get_densityfields(currentj, np, npt, pt, jc, par);
    int cdt = calcEBV(V, E, B, Ee, Be, npt, jc, par);
#pragma omp parallel sections
    {
#pragma omp section
        changedt(pt, cdt, par); /* change time step if E or B too big*/
#pragma omp section
#ifdef Uon_
        // cout << "calculate the total potential energy U\n";
        //                  timer.mark();
        calcU(V, E, B, pt, par);
        //                 cout << "U: " << timer.elapsed() << "s, ";
#endif
        save_files(i_time, t, np, currentj, V, E, B, pt, par);
        log_entry(0, 0, cdt, total_ncalc, t, par); // Write everything to log
#pragma omp section
        calc_trilin_constants(E, Ea, par);
#pragma omp section
        calc_trilin_constants(B, Ba, par);
    }
#pragma omp barrier

    cout << "print data: " << timer.elapsed() << "s (no. of electron time steps calculated: " << 0 << ")\n";

    for (i_time = 1; i_time < ndatapoints; i_time++)
    {
        for (int ntime = 0; ntime < nc; ntime++)
        {
            timer.mark();                                    // For timestep
            timer.mark();                                    // Work out motion
            tnp(Ea[0][0][0][0], Ba[0][0][0][0], 0, pt, par); //  calculate the next position par->ncalcp[p] times
            for (int p = 0; p < 2; ++p)
                total_ncalc[p] += par->ncalcp[p];
            cout << "motion: " << timer.elapsed() << "s, ";
            t += par->dt[0] * par->ncalcp[0];

            //  find number of particle and current density fields
            timer.mark();
            get_densityfields(currentj, np, npt, pt, jc, par);
            cout << "density: " << timer.elapsed() << "s, ";

            timer.mark();
            // set externally applied fields this is inside time loop so we can set time varying E and B field
            // calcEeBe(Ee,Be,t); // find E field must work out every i,j,k depends on charge in every other cell
            int cdt = calcEBV(V, E, B, Ee, Be, npt, jc, par);
            cout << "EBV: " << timer.elapsed() << "s, ";

            // calculate constants for each cell for trilinear interpolation
            timer.mark();
#pragma omp parallel sections
            {
#pragma omp section
                changedt(pt, cdt, par); // cout<<"change_dt done"<<endl;
#pragma omp section
                calc_trilin_constants(E, Ea, par);
#pragma omp section
                calc_trilin_constants(B, Ba, par);
            }
#pragma omp barrier
            cout << "trilin, calcU ... :  " << timer.elapsed() << "s\n";
            cout << i_time << "." << ntime << " t = " << t << "(compute_time = " << timer.elapsed() << "s) : ";
        }
        // print out all files for paraview

#ifdef Uon_
        // cout << "calculate the total potential energy U\n";
        // timer.mark();// calculate the total potential energy U
        calcU(V, E, B, pt, par); // cout << "U: " << timer.elapsed() << "s, ";
#endif
        timer.mark();
        save_files(i_time, t, np, currentj, V, E, B, pt, par);
        log_entry(i_time, 0, cdt, total_ncalc, t, par); // cout<<"log entry done"<<endl;
        cout << "print data: " << timer.elapsed() << "s (no. of electron time steps calculated: " << total_ncalc[0] << ")\n";
    }
    cout << "Overall execution time: " << timer.elapsed() << "s";
    logger.close();
    // info_file.close();
    return 0;
}
