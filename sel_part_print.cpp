#include "include/traj.h"
void sel_part_print(float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd],
                    float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd],
                    float posp[2][n_output_part][3], float KE[2][n_output_part],
                    int m[2][n_partd], par *par)
{
#pragma omp parallel num_threads(2)
    {
        int p = omp_get_thread_num();
        int nprtd = floor(par->n_part[p] / n_output_part);
#pragma omp parallel for simd
        // #pragma omp distribute parallel for simd
        for (int nprt = 0; nprt < n_output_part; nprt++)
        {
            int n = nprt * nprtd + rand() % nprtd;
            if (nprtd == 0 && n >= par->n_part[p])
            {
                KE[p][nprt] = 0;
                posp[p][nprt][0] = 0;
                posp[p][nprt][1] = 0;
                posp[p][nprt][2] = 0;
                continue;
            }
            float dpos, dpos2 = 0;
            dpos = (pos1x[p][n] - pos0x[p][n]);
            dpos *= dpos;
            dpos2 += dpos;
            dpos = (pos1y[p][n] - pos0y[p][n]);
            dpos *= dpos;
            dpos2 += dpos;
            dpos = (pos1z[p][n] - pos0z[p][n]);
            dpos *= dpos;
            dpos2 += dpos;
            KE[p][nprt] = 0.5 * m[p][n] * (dpos2) / (e_charge_mass * par->dt[p] * par->dt[p]);
            // in units of eV
            posp[p][nprt][0] = pos0x[p][n];
            posp[p][nprt][1] = pos0y[p][n];
            posp[p][nprt][2] = pos0z[p][n];
        }
    }
}