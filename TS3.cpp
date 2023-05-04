/* TS3.cpp
This contains the main loop for the program. Most of the initialization occurs here, and time steps are iterated through.
For settings (as to what to calculate, eg. E / B field, E / B force) go to the defines in include/traj.h
*/
#include "include/traj.h"
// sphere
int ncalc[2] = {md_me * 1, 1};
int n_part[3] = {n_parte, n_partd, n_parte + n_partd}; // 0,number of "super" electrons, electron +deuteriom ions, total
unsigned int n_space_div[3] = {n_space_divx, n_space_divy, n_space_divz};
unsigned int n_space_div2[3] = {n_space_divx2, n_space_divy2, n_space_divz2};
par par1;
par *par = &par1;
string outpath;
int main()
{
    omp_set_nested(true);
    // Fast printing
    cin.tie(NULL);
    // ios_base::sync_with_stdio(false);
    try
    {
        if (!std::filesystem::create_directory(outpath1))
            outpath = outpath1;
        else if (!std::filesystem::create_directory(outpath2))
            outpath = outpath2;
    }
    catch (const std::filesystem::__cxx11::filesystem_error &e)
    {
        std::cerr << "Error creating output directory: " << e.what() << '\n';
        try
        {
            if (!std::filesystem::create_directory(outpath2))
                outpath = outpath2;
        }
        catch (const std::filesystem::__cxx11::filesystem_error &e)
        {
            std::cerr << "Error creating output directory: " << e.what() << '\n';
        }
    }
    std::cout << "Output dir: " << outpath << "\n";

    timer.mark(); // Yes, 3 time marks. The first is for the overall program dt
    timer.mark(); // The second is for compute_d_time
    timer.mark(); // The third is for start up dt

    // omp_set_num_threads(nthreads);
    nthreads = omp_get_max_threads();
    double t = 0;

    const unsigned int n_cells = n_space_divx * n_space_divy * n_space_divz;
    cout << "(unsigned int) ((int)(-2.5f))" << (unsigned int)((int)(-2.5f)) << endl;
    // position of particle and velocity: stored as 2 positions at slightly different times
    /** CL: Ensure that pos0/1.. contain multiple of 64 bytes, ie. multiple of 16 floats **/
    auto *pos0x = reinterpret_cast<float(&)[2][n_partd]>(*((float *)_aligned_malloc(sizeof(float) * n_partd * 2, 4096))); // new float[2][n_partd];
    auto *pos0y = reinterpret_cast<float(&)[2][n_partd]>(*((float *)_aligned_malloc(sizeof(float) * n_partd * 2, 4096))); // new float[2][n_partd];
    auto *pos0z = reinterpret_cast<float(&)[2][n_partd]>(*((float *)_aligned_malloc(sizeof(float) * n_partd * 2, 4096))); // new float[2][n_partd];
    auto *pos1x = reinterpret_cast<float(&)[2][n_partd]>(*((float *)_aligned_malloc(sizeof(float) * n_partd * 2, 4096))); // new float[2][n_partd];
    auto *pos1y = reinterpret_cast<float(&)[2][n_partd]>(*((float *)_aligned_malloc(sizeof(float) * n_partd * 2, 4096))); // new float[2][n_partd];
    auto *pos1z = reinterpret_cast<float(&)[2][n_partd]>(*((float *)_aligned_malloc(sizeof(float) * n_partd * 2, 4096))); // new float[2][n_partd];

    //    charge of particles
    auto *q = static_cast<int(*)[n_partd]>(_aligned_malloc(2 * n_partd * sizeof(int), alignment)); // charge of each particle +1 for H,D or T or -1 for electron can also be +2 for He for example
    auto *m = static_cast<int(*)[n_partd]>(_aligned_malloc(2 * n_partd * sizeof(int), alignment)); // mass of of each particle not really useful unless we want to simulate many different types of particles

    // reduced particle position dataset for printing/plotting
    auto *posp = new float[2][n_output_part][3];
    auto *KE = new float[2][n_output_part];

    /** CL: Ensure that Ea/Ba contain multiple of 64 bytes, ie. multiple of 16 floats **/
    auto *E = reinterpret_cast<float(&)[3][n_space_divz][n_space_divy][n_space_divx]>(*fftwf_alloc_real(3 * n_cells)); // selfgenerated E field
    auto *Ee = new float[3][n_space_divz][n_space_divy][n_space_divx];                                                 // External E field
    float *Ea1 = (float *)_aligned_malloc(sizeof(float) * n_cells * 3 * ncoeff, 4096);                                 // coefficients for Trilinear interpolation Electric field
    auto *Ea = reinterpret_cast<float(&)[n_space_divz][n_space_divy][n_space_divx][3][ncoeff]>(*Ea1);

    auto *B = reinterpret_cast<float(&)[3][n_space_divz][n_space_divy][n_space_divx]>(*fftwf_alloc_real(3 * n_cells)); // new float[3][n_space_divz][n_space_divy][n_space_divx];
    auto *Be = new float[3][n_space_divz][n_space_divy][n_space_divx];
    float *Ba1 = (float *)_aligned_malloc(sizeof(float) * n_cells * 3 * ncoeff, 4096); // coefficients for Trilinear interpolation Magnetic field
    auto *Ba = reinterpret_cast<float(&)[n_space_divz][n_space_divy][n_space_divx][3][ncoeff]>(*Ba1);

    auto *V = reinterpret_cast<float(&)[n_space_divz][n_space_divy][n_space_divx]>(*fftwf_alloc_real(n_cells));

    auto *np = static_cast<float(*)[n_space_divz][n_space_divy][n_space_divx]>(_aligned_malloc(2 * n_space_divz * n_space_divy * n_space_divx * sizeof(float), alignment));
    auto *npt = static_cast<float(*)[n_space_divy][n_space_divx]>(_aligned_malloc(n_space_divz * n_space_divy * n_space_divx * sizeof(float), alignment));
    auto *currentj = static_cast<float(*)[3][n_space_divz][n_space_divy][n_space_divx]>(_aligned_malloc(2 * 3 * n_space_divz * n_space_divy * n_space_divx * sizeof(float), alignment));
    auto *jc = static_cast<float(*)[n_space_divz][n_space_divy][n_space_divx]>(_aligned_malloc(3 * n_space_divz * n_space_divy * n_space_divx * sizeof(float), alignment));

    // float U[2] = {0, 0}; //{Electric PE, magnetic PE}

    ofstream E_file, B_file;

    log_headers();

    cout << std::scientific;
    cout.precision(1);
    cerr << std::scientific;
    cerr.precision(3);
    cout << "float size=" << sizeof(float) << ", "
         << "int32_t size=" << sizeof(int32_t) << ", "
         << "int size=" << sizeof(int) << endl;
    // const float coef = (float)qs[p] * e_charge_mass / (float)mp[p] * dt[p] * 0.5f;
    int total_ncalc[2] = {0, 0}; // particle 0 - electron, particle 1 deuteron
                                 //   float dt[2];
    cout << "Start up time = " << timer.replace() << "s\n";
#define generateRandom
#ifdef generateRandom
#ifdef sphere
    generate_rand_sphere(pos0x, pos0y, pos0z, pos1x, pos1y, pos1z, q, m, par);
#endif // sphere
#ifdef cylinder
    generate_rand_cylinder(pos0x, pos0y, pos0z, pos1x, pos1y, pos1z, q, m, par);
#endif // cylinder
#else
    generateParticles(a0, r0, qs, mp, pos0x, pos0y, pos0z, pos1x, pos1y, pos1z, q, m, nt);
    n_part[0] = abs(nt[0]);
    n_part[1] = abs(nt[1]);
    n_part[2] = n_part[0] + n_part[1];
#endif
    cout << "dt = {" << par->dt[0] << ", " << par->dt[1] << "}\n";
    // get limits and spacing of Field cells
    generateField(Ee, Be);

    cout << "Set initial random positions: " << timer.replace() << "s\n";
    /*
    float posL[3], posH[3], posL2[3], dd[3];
    // set spacing between cells
    for (int c = 0; c < 3; c++)
        dd[c] = a0;
    // set position of centers of the cells at extreme ends

    for (int c = 0; c < 3; c++)
    {
        posL[c] = -dd[c] * (n_space_div[c] - 1.0) / 2.0;
        posH[c] = dd[c] * (n_space_div[c] - 1.0) / 2.0;
        posL2[c] = -dd[c] * (n_space_div[c]);
        cout << posL[c] << "," << posH[c] << "," << dd[c] << endl;
    }
*/
    unsigned int ci[2] = {n_partd, 0};
    float cf[2] = {0, 0};
    fftwf_init_threads();
    cl_set_build_options(par);
    cl_start();

    // print initial conditions
    {
        E_file.open("info.csv");
        E_file << "Data Origin," << par->posL[0] << "," << par->posL[1] << "," << par->posL[0] << endl;
        E_file << "Data Spacing," << par->dd[0] << "," << par->dd[1] << "," << par->dd[2] << endl;
        E_file << "Data extent x, 0," << n_space - 1 << endl;
        E_file << "Data extent y, 0," << n_space - 1 << endl;
        E_file << "Data extent z, 0," << n_space - 1 << endl;
        //       E_file << "electron Temp = ," << Temp[0] << ",K" << endl;
        E_file << "Maximum expected B = ," << par->Bmax << endl;
        E_file << "time step between prints = ," << par->dt[0] * ncalc[0] * nc << ",s" << endl;
        E_file << "time step between EBcalc = ," << par->dt[0] * ncalc[0] << ",s" << endl;
        E_file << "dt =," << par->dt[0] << ",s" << endl;
        E_file << "cell size =," << a0 << ",m" << endl;
        E_file << "number of particles per cell = ," << n_partd / (n_space * n_space * n_space) << endl;
        E_file.close();
    }

    int i_time = 0;
    get_densityfields(currentj, np, npt, pos1x, pos1y, pos1z, pos0x, pos0y, pos0z, q, jc, par);
    calcEBV(V, E, B, Ee, Be, npt, jc, par);
    //   cout << "calc trilin constants\n";
    calc_trilin_constants(E, Ea, par);
    calc_trilin_constants(B, Ba, par);
#ifdef Uon_
    //   cout << "calcU\n";
    calcU(V, E, B, pos1x, pos1y, pos1z, q, par);
#endif
    //    cout << i_time << "." << 0 << " (compute_time = " << timer.elapsed() << "s): ";
    //    cout << "dt = {" << par->dt[0] << " " << par->dt[1] << "}, t_sim = " << t << " s" << ", ne = " << nt[0] << ", ni = " << nt[1];
    //    cout << "KEtot e = " << KEtot[0] << ", KEtot i = " << KEtot[1] << ", Eele = " << U[0] << ", Emag = " << U[1] << ", Etot = " << KEtot[0] + KEtot[1] + U[0] + U[1] << " eV\n";
    sel_part_print(n_part, pos1x, pos1y, pos1z, pos0x, pos0y, pos0z, posp, KE, m, par);
    save_files(i_time, t, np, currentj, V, E, B, KE, posp, par);
    cout << "print data: " << timer.elapsed() << "s (no. of electron time steps calculated: " << 0 << ")\n";

    // Write everything to log
    log_entry(i_time, 0, 0, total_ncalc, t, par);

    for (i_time = 1; i_time < ndatapoints; i_time++)
    {

        for (int ntime = 0; ntime < nc; ntime++)
        {

            timer.mark(); // For timestep
            // Work out motion
            timer.mark();
#pragma omp parallel num_threads(2)
            {
                int p = omp_get_thread_num();
#ifdef BFon_
                par->Bcoef = (float)qs[p] * e_charge_mass / (float)mp[p] * par->dt[p] * 0.5f;
#else
                par->Bcoef = 0;
#endif
#ifdef EFon_
                par->Ecoef = par->Bcoef * par->dt[p]; // multiply by dt because of the later portion of cl code
#else
                par->Ecoef = 0;
#endif
                par->ncalcp = ncalc[p];
                par->n_partp = n_part[p];
                // cout << p << " Bconst=" <<par->Bcoef  << ", Econst=" << par->Ecoef << endl;
                tnp(Ea1, Ba1, pos0x[p], pos0y[p], pos0z[p], pos1x[p], pos1y[p], pos1z[p],p, par); //  calculate the next position ncalc[p] times
                total_ncalc[p] += ncalc[p];
            }
#pragma omp barrier
            cout << "motion: " << timer.elapsed() << "s, ";
            t += par->dt[0] * ncalc[0];
            //  find number of particle and current density fields
            timer.mark();
            get_densityfields(currentj, np, npt, pos1x, pos1y, pos1z, pos0x, pos0y, pos0z, q, jc, par);
            cout << "density: " << timer.elapsed() << "s, ";
            save_hist(i_time, t, pos0x, pos0y, pos0z, pos1x, pos1y, pos1z, par);
            // find E field must work out every i,j,k depends on charge in every other cell
            timer.mark();
            // set externally applied fields this is inside time loop so we can set time varying E and B field
            // calcEeBe(Ee,Be,t);
            int cdt = calcEBV(V, E, B, Ee, Be, npt, jc, par);
            cout << "EBV: " << timer.elapsed() << "s, ";

            // calculate constants for each cell for trilinear interpolation
            timer.mark();
#pragma omp parallel sections
            {
#pragma omp section
                {
                    changedt(pos0x, pos0y, pos0z, pos1x, pos1y, pos1z, n_part, cdt, par); /* change time step if E or B too big*/
                }
#pragma omp section
                {
#ifdef Uon_
                    // calculate the total potential energy U
                    //        cout << "calculate the total potential energy U\n";
                    //                  timer.mark();
                    calcU(V, E, B, pos1x, pos1y, pos1z, q, par);
                    //                 cout << "U: " << timer.elapsed() << "s, ";
#endif
                }
#pragma omp section
                {
                    calc_trilin_constants(E, Ea, par);
                }

#pragma omp section
                {
                    calc_trilin_constants(B, Ba, par);
                }

#pragma omp section
                {
                    log_entry(i_time, ntime, cdt, total_ncalc, t, par);
                }
            }
#pragma omp barrier
            cout << "trilin const &calcU :  " << timer.elapsed() << "s\n";
            cout << i_time << "." << ntime << " t = " << t << "(compute_time = " << timer.elapsed() << "s) : ";
            // if (cdt)  cout << "dtchanged\n";
            //  cout << "dt = {" << par->dt[0] << " " << par->dt[1] << "}, t_sim = " << t << " s"<< ", ne = " << nt[0] << ", ni = " << nt[1];
            //  cout << "\nKEtot e = " << KEtot[0] << ", KEtot i = " << KEtot[1] << ", Eele = " << U[0] << ", Emag = " << U[1] << ", Etot = " << KEtot[0] + KEtot[1] + U[0] + U[1] << " eV\n";
        }

        // print out all files for paraview
        timer.mark();
        sel_part_print(n_part, pos1x, pos1y, pos1z, pos0x, pos0y, pos0z, posp, KE, m, par);
        save_files(i_time, t, np, currentj, V, E, B, KE, posp, par);
        cout << "print data: " << timer.elapsed() << "s (no. of electron time steps calculated: " << total_ncalc[0] << ")\n";
    }
    cout << "Overall execution time: " << timer.elapsed() << "s";
    logger.close();
    return 0;
}
