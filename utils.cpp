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
    // print initial conditions
    {
        ofstream E_file;
        E_file.open("info.csv");
        E_file << "Data Origin," << par->posL[0] << "," << par->posL[1] << "," << par->posL[0] << endl;
        E_file << "Data Spacing," << par->dd[0] << "," << par->dd[1] << "," << par->dd[2] << endl;
        E_file << "Data extent x, 0," << n_space - 1 << endl;
        E_file << "Data extent y, 0," << n_space - 1 << endl;
        E_file << "Data extent z, 0," << n_space - 1 << endl;
        //       E_file << "electron Temp = ," << Temp[0] << ",K" << endl;
        E_file << "Maximum expected B = ," << par->Bmax << endl;
        E_file << "time step between prints = ," << par->dt[0] * par->ncalcp[0] * nc << ",s" << endl;
        E_file << "time step between EBcalc = ," << par->dt[0] * par->ncalcp[0] << ",s" << endl;
        E_file << "dt =," << par->dt[0] << ",s" << endl;
        E_file << "cell size =," << a0 << ",m" << endl;
        E_file << "number of particles per cell = ," << n_partd / (n_space * n_space * n_space) << endl;
        E_file.close();
    }
}