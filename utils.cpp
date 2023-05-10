#include "include/traj.h"

void id_to_cell(int id, int *x, int *y, int *z)
{
    constexpr size_t xy = n_space_divx * n_space_divy;
    *z = id / xy;
    id = id % xy;
    *y = id / n_space_divx;
    *x = id % n_space_divx;
}

void Time::mark()
{
    marks.push_back(chrono::high_resolution_clock::now());
}

float Time::elapsed()
{
    unsigned long long time = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - marks.back()).count();
    marks.pop_back();
    return (float)time * 1e-6;
}

// Get the same result as elapsed, but also insert the current time point back in
float Time::replace()
{
    auto now = chrono::high_resolution_clock::now();
    auto back = marks.back();
    unsigned long long time = chrono::duration_cast<chrono::microseconds>(now - back).count();
    back = now;
    return (float)time * 1e-6;
}

Log::Log()
{
    if (!log_file.is_open())
        log_file.open("log.csv");
    log_file << setprecision(5);
}

void Log::newline()
{
    log_file << "\n";
    log_file.flush();
    firstEntry = true;
}
void Log::close()
{
    log_file.close();
}

void log_headers()
{
    logger.write("t_large");
    logger.write("t_small");
    logger.write("dt_ch");
    logger.write("nc_ele");
    logger.write("nc_deut");
    logger.write("dt_ele_fs");
    logger.write("dt_deut_fs");
    logger.write("t_sim_ps");
    logger.write("ne");
    logger.write("ni");
    logger.write("KEt_e");
    logger.write("KEt_d");
    logger.write("Ele_pot");
    logger.write("Mag_pot");
    logger.write("E_tot");
    logger.write("Emax");
    logger.write("Bmax");
    logger.newline();
}

void log_entry(int i_time, int ntime, int cdt, int total_ncalc[2], double t, par *par)
{
    logger.write(i_time);
    logger.write(ntime);
    logger.write(cdt);
    logger.write(total_ncalc[0]);
    logger.write(total_ncalc[1]);
    logger.write(par->dt[0] * 1e15); // in fs
    logger.write(par->dt[1] * 1e15);
    logger.write(t * 1e12); // in ps
    logger.write(par->nt[0]);
    logger.write(par->nt[1]);
    logger.write(par->KEtot[0]);
    logger.write(par->KEtot[1]);
    logger.write(par->UB);
    logger.write(par->UE);
    logger.write(par->KEtot[0] + par->KEtot[1] + par->UB + par->UE);
    logger.write(par->Emax);
    logger.write(par->Bmax);
    logger.newline();
}
float maxvalf(float *data_1d, int n)
{
    float max = 0;
#pragma omp parallel for reduction(max : max)
    for (unsigned int i = 0; i < n; ++i)
    {
        float absVal = fabs(data_1d[i]);
        max = (absVal > max) ? absVal : max; // use the ternary operator to update the maximum
    }
    return max;
}

void info(par *par)
{
    info_file.open("info.csv");
    omp_set_nested(true);
    nthreads = omp_get_max_threads(); // omp_set_num_threads(nthreads);

    cin.tie(NULL); // Fast printing
    ios_base::sync_with_stdio(false);
    try
    {
        if (!std::filesystem::create_directory(outpath1))
            par->outpath = outpath1;
        else if (!std::filesystem::create_directory(outpath2))
            par->outpath = outpath2;
    }
    catch (const std::filesystem::__cxx11::filesystem_error &e)
    {
        std::cerr << "Error creating output directory: " << e.what() << '\n';
        try
        {
            if (!std::filesystem::create_directory(outpath2))
                par->outpath = outpath2;
        }
        catch (const std::filesystem::__cxx11::filesystem_error &e)
        {
            std::cerr << "Error creating output directory: " << e.what() << '\n';
        }
    }
    info_file << "Output dir: " << par->outpath << "\n";
    cl_set_build_options(par);
    cl_start(par);
    // print initial conditions
    {
        info_file << "float size=" << sizeof(float) << ", "
                  << "int32_t size=" << sizeof(int32_t) << ", "
                  << "int size=" << sizeof(int) << "(unsigned int) ((int)(-2.5f))" << (unsigned int)((int)(-2.5f)) << endl;
        info_file << "omp_get_max_threads()= " << omp_get_max_threads() << endl;
        info_file << "Data Origin," << par->posL[0] << "," << par->posL[1] << "," << par->posL[0] << endl;
        info_file << "Data Spacing," << par->dd[0] << "," << par->dd[1] << "," << par->dd[2] << endl;
        info_file << "Data extent x, 0," << n_space - 1 << endl;
        info_file << "Data extent y, 0," << n_space - 1 << endl;
        info_file << "Data extent z, 0," << n_space - 1 << endl;
        info_file << "electron Temp+e = ," << Temp_e << ",K" << endl;
        info_file << "Maximum expected B = ," << par->Bmax << endl;
        info_file << "time step between prints = ," << par->dt[0] * par->ncalcp[0] * nc << ",s" << endl;
        info_file << "time step between EBcalc = ," << par->dt[0] * par->ncalcp[0] << ",s" << endl;
        info_file << "dt_e = ," << par->dt[0] << ",s" << endl;
        info_file << "dt_i = ," << par->dt[1] << ",s" << endl;
        info_file << "cell size =," << a0 << ",m" << endl;
        info_file << "number of particles per cell = ," << n_partd / (n_space * n_space * n_space) << endl;
        info_file << "time for electrons to leave box = ," << n_space * a0 / sqrt(2 * kb * Temp_e / e_mass) << ",s" << endl;
        info_file << "time for ions to leave box = ," << n_space * a0 * md_me / sqrt(2 * kb * Temp_d / e_mass) << ",s" << endl;
    }
}
particles *alloc_particles(par *par)
{
    auto *s = (particles *)malloc(sizeof(particles)); /*
     //[pos0,pos1][x,y,z][electrons,ions][n_partd]
     auto *pos0 = reinterpret_cast<float(&)[2][3][2][par->n_part[0]]>(*((float *)_aligned_malloc(sizeof(float) * par->n_part[0] * 2 * 3 * 2, par->cl_align)));
     // auto *pos1 = reinterpret_cast<float(&)[3][2][par->n_part[0]]>(*((float *)_aligned_malloc(sizeof(float) * par->n_part[0] * 2 * 3, par->cl_align)));

     s->pos0x = &pos0[0][0][0][0];
     s->pos0y = &pos0[0][1][0][0];
     s->pos0z = &pos0[0][2][0][0];
     s->pos1x = &pos0[1][0][0][0];
     s->pos1y = &pos0[1][1][0][0];
     s->pos1x = &pos0[1][2][0][0];
     auto *pos0y = reinterpret_cast<float(&)[2][par->n_part[0]]>(*(float *)(pos0[1]));
         auto *pos0z = reinterpret_cast<float(&)[2][par->n_part[0]]>(*(float *)(pos0[2]));
         auto *pos1x = reinterpret_cast<float(&)[2][par->n_part[0]]>(*(float *)(pos1[0]));
         auto *pos1y = reinterpret_cast<float(&)[2][par->n_part[0]]>(*(float *)(pos1[1]));
         auto *pos1z = reinterpret_cast<float(&)[2][par->n_part[0]]>(*(float *)(pos1[2]));*/
    return s;
}
