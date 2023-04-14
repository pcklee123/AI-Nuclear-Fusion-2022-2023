#include "include/traj.h"

void get_densityfields(float currentj[2][3][n_space_divz][n_space_divy][n_space_divx],
                       float np[2][n_space_divz][n_space_divy][n_space_divx],
                       float npt[n_space_divz][n_space_divy][n_space_divx],
                       int nt[2], float KEtot[2], float posL[3], float posH[3], float dd[3],
                       float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd],
                       float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd],
                       int q[2][n_partd], float dt[2], int n_part[3],
                       float jc[3][n_space_divz][n_space_divy][n_space_divx])
{
    // find number of particle and current density fields
    // set fields=0 in preparation
    unsigned int n;

    fill(reinterpret_cast<float *>(currentj), reinterpret_cast<float *>(currentj) + n_cells * 6, 0.f);
    fill(reinterpret_cast<float *>(np), reinterpret_cast<float *>(np) + n_cells * 2, 0); // Could split into threads.

    nt[0] = 0;
    nt[1] = 0;
    KEtot[0] = 0;
    KEtot[1] = 0;
    // set limits beyond which particle is considered as "lost"
    static float xld = posL[0] + dd[0] * 1;
    static float yld = posL[1] + dd[1] * 1;
    static float zld = posL[2] + dd[2] * 1;
    static float xhd = posH[0] - dd[0] * 1;
    static float yhd = posH[1] - dd[1] * 1;
    static float zhd = posH[2] - dd[2] * 1;
    static float ddi[3] = {1.f / dd[0], 1.f / dd[1], 1.f / dd[2]};
    static float dti[2] = {1.f / dt[0], 1.f / dt[1]};
   // static float v[2][3][n_parte];
   // static unsigned int ii[2][3][n_parte];
    static auto ii = new unsigned int[2][3][n_parte];
    static auto v = new float[2][3][n_parte];
        cout << "get_density_start\n";
#pragma omp parallel num_threads(2)
    {
        int p = omp_get_thread_num();
        int remainder = n_part[p] % nthreads;
        int particles_per_thread = n_part[p] / nthreads;
#pragma omp parallel num_threads(nthreads)
        {
            int thread_id = omp_get_thread_num();
            int start_index = thread_id * particles_per_thread;
            int end_index = (thread_id + 1) * particles_per_thread;
            if (thread_id == nthreads - 1)
            {
                end_index += remainder;
            }
            //          cout << "thread " << thread_id << endl;
            for (int n = start_index; n < end_index; n++)
            {
                // replace out of bounds particles and reduce the number of particles.
                if ((pos1x[p][n] <= xld) | (pos1y[p][n] <= yld) | (pos1z[p][n] <= zld) |
                    (pos1x[p][n] >= xhd) | (pos1y[p][n] >= yhd) | (pos1z[p][n] >= zhd) |
                    (pos0x[p][n] <= xld) | (pos0y[p][n] <= yld) | (pos0z[p][n] <= zld) |
                    (pos0x[p][n] >= xhd) | (pos0y[p][n] >= yhd) | (pos0z[p][n] >= zhd))
                {
#pragma omp atomic
                    n_part[p]--;
                    int last = n_part[p];
                    pos0x[p][n] = pos0x[p][last];
                    pos0y[p][n] = pos0y[p][last];
                    pos0z[p][n] = pos0z[p][last];
                    pos1x[p][n] = pos1x[p][last];
                    pos1y[p][n] = pos1y[p][last];
                    pos1z[p][n] = pos1z[p][last];
                    q[p][n] = q[p][last];
                    q[p][last] = 0;
                    n--;
                }
            }
        }
    }
#pragma omp barrier
    cout << "get_density_checked out of bounds\n";
#pragma omp parallel num_threads(2)
    {
        int p = omp_get_thread_num();
#pragma omp parallel for simd num_threads(nthreads)
        for (int n = 0; n < n_part[p]; ++n)
        {
            v[p][0][n] = (pos1x[p][n] - pos0x[p][n]) * dti[p];
            v[p][1][n] = (pos1y[p][n] - pos0y[p][n]) * dti[p];
            v[p][2][n] = (pos1z[p][n] - pos0z[p][n]) * dti[p];
        }
#pragma omp parallel for simd num_threads(nthreads)
        for (int n = 0; n < n_part[p]; ++n)
        {
            KEtot[p] += v[p][0][n] * v[p][0][n] + v[p][1][n] * v[p][1][n] + v[p][2][n] * v[p][2][n];
        }
        KEtot[p] *= 0.5 * mp[p] / (e_charge_mass)*r_part_spart; // as if these particles were actually samples of the greater thing
//      cout <<maxk <<",";
#pragma omp parallel for simd num_threads(nthreads)
        for (int n = 0; n < n_part[p]; ++n)
        {
            v[p][0][n] *= q[p][n];
            v[p][1][n] *= q[p][n];
            v[p][2][n] *= q[p][n];
        }
#pragma omp parallel for simd num_threads(nthreads)
        for (int n = 0; n < n_part[p]; ++n)
        {
            ii[p][0][n] = (unsigned int)((pos1x[p][n] - posL[0]) * ddi[0]);
            ii[p][1][n] = (unsigned int)((pos1y[p][n] - posL[1]) * ddi[1]);
            ii[p][2][n] = (unsigned int)((pos1z[p][n] - posL[2]) * ddi[2]);
        }

#pragma omp parallel num_threads(nthreads)
        {
            int thread_id = omp_get_thread_num();
            int remainder = n_part[p] % nthreads;
            int particles_per_thread = n_part[p] / nthreads;
            int start_index = thread_id * particles_per_thread;
            int end_index = (thread_id + 1) * particles_per_thread;
            if (thread_id == nthreads - 1)
                end_index += remainder;
            //      cout << "npp" << p << n_part[p] << endl;
            for (int n = start_index; n < end_index; n++)
            {
                unsigned int i = ii[p][0][n], j = ii[p][1][n], k = ii[p][2][n];
                np[p][k][j][i] += q[p][n];
                nt[p] += q[p][n];
                currentj[p][0][k][j][i] += v[p][0][n];
                currentj[p][1][k][j][i] += v[p][1][n];
                currentj[p][2][k][j][i] += v[p][2][n];
            }
        }
    }
#pragma omp barrier
    cout << "get_density_almost done\n";

#pragma omp parallel for simd num_threads(nthreads)
    for (unsigned int i = 0; i < n_cells * 3; i++)
    {
        (reinterpret_cast<float *>(jc))[i] = (reinterpret_cast<float *>(currentj[0]))[i] + (reinterpret_cast<float *>(currentj[1]))[i];
    }
#pragma omp parallel for simd num_threads(nthreads)
    for (unsigned int i = 0; i < n_cells; i++)
    {
        (reinterpret_cast<float *>(npt))[i] = (reinterpret_cast<float *>(np[0]))[i] + (reinterpret_cast<float *>(np[1]))[i];
    }
}