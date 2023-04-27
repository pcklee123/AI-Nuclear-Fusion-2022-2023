#include "include/traj.h"
// Interpolate the value at a given point

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

    nt[0] = 0;
    nt[1] = 0;
    KEtot[0] = 0;
    KEtot[1] = 0;
    // set limits beyond which particle is considered as "lost"
    static const float ddi[3] = {1.f / dd[0], 1.f / dd[1], 1.f / dd[2]}; // precalculate reciprocals
    const float dti[2] = {1.f / dt[0], 1.f / dt[1]};

    // cell indices for each particle [2][3][n_parte]
    static auto *ii = static_cast<unsigned int(*)[3][n_parte]>(_aligned_malloc(2 * 3 * n_parte * sizeof(unsigned int), alignment));
    // particle velocity array [2][3][n_parte]
    static auto *v = static_cast<float(*)[3][n_parte]>(_aligned_malloc(2 * 3 * n_parte * sizeof(float), alignment));
    // particle offsets array [2][3][n_parte]
    static auto *offset = static_cast<float(*)[3][n_parte]>(_aligned_malloc(2 * 3 * n_parte * sizeof(float), alignment));

    // center of charge field arrays [2-particle type][3 pos][z][y][x]
    static auto *np_center = static_cast<float(*)[n_space_divz][n_space_divy][n_space_divx][3]>(_aligned_malloc(2 * 3 * n_space_divz * n_space_divy * n_space_divx * sizeof(float), alignment));
    // center of current field arrays [2][3-pos][3-current component][z][y][x]
    static auto *jc2 = static_cast<float(*)[2][3][n_space_divz][n_space_divy][n_space_divx]>(_aligned_malloc(2 * 2 * 3 * n_space_divz * n_space_divy * n_space_divx * sizeof(float), alignment));
    static auto *jc_center = static_cast<float(*)[2][3][n_space_divz][n_space_divy][n_space_divx][3]>(_aligned_malloc(2 * 2 * 3 * 3 * n_space_divz * n_space_divy * n_space_divx * sizeof(float), alignment));
    //   static auto *jc_center = static_cast<float(*)[3][n_space_divz][n_space_divy][n_space_divx][3]>(_aligned_malloc(2 * 3 * 3 * n_space_divz * n_space_divy * n_space_divx * sizeof(float), alignment));

    // set fields=0 in preparation// Could split into threads
    fill(reinterpret_cast<float *>(np), reinterpret_cast<float *>(np) + n_cells * 2, 0.f);
    //   fill(reinterpret_cast<float *>(currentj), reinterpret_cast<float *>(currentj) + n_cells * 2 * 3, 0.f);

    fill(reinterpret_cast<float *>(np_center), reinterpret_cast<float *>(np_center) + n_cells * 3 * 2, 0.f);
    fill(reinterpret_cast<float *>(jc2), reinterpret_cast<float *>(jc2) + n_cells * 2 * 2 * 3, 0.f);
    fill(reinterpret_cast<float *>(jc_center), reinterpret_cast<float *>(jc_center) + n_cells * 2 * 2 * 3 * 3, 0.f);

    static auto oblist = new unsigned int[2][n_parte]; // list of out of bound particles
    static auto iblist = new unsigned int[2][n_parte];
    int nob[2]; // number of particles out of bounds
    int nib[2]; // number of particles within bounds
//   cout << "get_density_start\n";
// remove out of bounds points and get x,y,z index of each particle
#pragma omp parallel num_threads(2)
    {
        int p = omp_get_thread_num();
#pragma omp parallel for simd num_threads(nthreads)
        for (unsigned int n = 0; n < n_part[p]; ++n) // get cell indices (x,y,z) a particle belongs to
        {
            ii[p][0][n] = (int)roundf((pos1x[p][n] - posL[0]) * ddi[0]);
            ii[p][1][n] = (int)roundf((pos1y[p][n] - posL[1]) * ddi[1]);
            ii[p][2][n] = (int)roundf((pos1z[p][n] - posL[2]) * ddi[2]);
            offset[p][0][n] = (pos1x[p][n] - posL[0]) * ddi[0] - (float)(ii[p][0][n]);
            offset[p][1][n] = (pos1y[p][n] - posL[1]) * ddi[1] - (float)(ii[p][1][n]);
            offset[p][2][n] = (pos1z[p][n] - posL[2]) * ddi[2] - (float)(ii[p][2][n]);
            //        if ((pos1x[p][n] - posL[0]) * ddi[0] < 0)
            //          cout << ii[p][0][n] << " ";
        }
    }
#pragma omp barrier
#pragma omp parallel num_threads(2)
    {
        int p = omp_get_thread_num();
        nob[p] = 0;
        nib[p] = 0;
        int nzm = 0;
        int nzp = 0;
        for (unsigned int n = 0; n < n_part[p]; ++n)
        { // particle on 1st and last cell must be rejected so that indices [k-1][j-1][i-1] ..  [k+1][j+1][i+1] are OK.
//  if ((ii[p][0][n] > (n_space_divx - 1)) || (ii[p][1][n] > (n_space_divy - 1)) || (ii[p][2][n] > (n_space_divz - 1)))
#ifdef cylinder // rollover particles in z direction
            if (ii[p][2][n] == 0)
            {
                pos0z[p][n] += (n_space_divz - 2) * dd[2];
                pos1z[p][n] += (n_space_divz - 2) * dd[2];
                ii[p][2][n] = n_space_divz - 2;
                nzm++;
            }
            if (ii[p][2][n] == n_space_divz - 1)
            {
                pos0z[p][n] -= (n_space_divz - 2) * dd[2];
                pos1z[p][n] -= (n_space_divz - 2) * dd[2];
                ii[p][2][n] = 1;
                nzp++;
            }

            // hit wall replace with stationary particle
            if (ii[p][0][n] == 0)
            {
                pos1x[p][n] += dd[0];
                pos0x[p][n] = pos1x[p][n];
                ii[p][0][n]++;
            }
            if (ii[p][0][n] == n_space_divx - 1)
            {
                pos1x[p][n] -= dd[0];
                pos0x[p][n] = pos1x[p][n];
                ii[p][0][n]--;
            }
            if (ii[p][1][n] == 0)
            {
                pos1y[p][n] += dd[1];
                pos0y[p][n] = pos1y[p][n];
                ii[p][1][n]++;
            }
            if (ii[p][1][n] == n_space_divy - 1)
            {
                pos1y[p][n] -= dd[1];
                pos0y[p][n] = pos1y[p][n];
                ii[p][1][n]--;
            }
#endif
#ifdef sphere
            // hit wall replace with stationary particle
            if (ii[p][0][n] == 0)
            {
                pos1x[p][n] += dd[0];
                pos0x[p][n] = pos1x[p][n];
                ii[p][0][n]++;
            }
            if (ii[p][0][n] == n_space_divx - 1)
            {
                pos1x[p][n] -= dd[0];
                pos0x[p][n] = pos1x[p][n];
                ii[p][0][n]--;
            }
            if (ii[p][1][n] == 0)
            {
                pos1y[p][n] += dd[1];
                pos0y[p][n] = pos1y[p][n];
                ii[p][1][n]++;
            }
            if (ii[p][0][n] == n_space_divy - 1)
            {
                pos1y[p][n] -= dd[1];
                pos0y[p][n] = pos1y[p][n];
                ii[p][1][n]--;
            }
            if (ii[p][2][n] == 0)
            {
                pos1z[p][n] += dd[2];
                pos0z[p][n] = pos1z[p][n];
                ii[p][2][n]++;
            }
            if (ii[p][0][n] == n_space_divz - 1)
            {
                pos1z[p][n] -= dd[2];
                pos0z[p][n] = pos1z[p][n];
                ii[p][2][n]--;
            }
#endif
            if ((ii[p][0][n] > (n_space_divx - 2)) || (ii[p][1][n] > (n_space_divy - 2)) || (ii[p][2][n] > (n_space_divz - 2)) || (ii[p][0][n] < 1) || (ii[p][1][n] < 1) || (ii[p][2][n] < 1))
            {
                oblist[p][nob[p]] = n;
                nob[p]++;
            }
            else
            {
                iblist[p][nob[p]] = n;
                nib[p]++;
            }
        }
        cout << "nzp " << nzp << endl;
        cout << "nzm " << nzm << endl;

        for (unsigned int n = 0; n < nob[p]; ++n)
        {
            n_part[p]--;
            nib[p]--;
            int last = iblist[p][nib[p]];
            pos0x[p][oblist[p][n]] = pos0x[p][last];
            pos0y[p][oblist[p][n]] = pos0y[p][last];
            pos0z[p][oblist[p][n]] = pos0z[p][last];
            pos1x[p][oblist[p][n]] = pos1x[p][last];
            pos1y[p][oblist[p][n]] = pos1y[p][last];
            pos1z[p][oblist[p][n]] = pos1z[p][last];
            ii[p][0][oblist[p][n]] = ii[p][0][last];
            ii[p][1][oblist[p][n]] = ii[p][1][last];
            ii[p][2][oblist[p][n]] = ii[p][2][last];
            q[p][n] = q[p][last];
            q[p][last] = 0;
        }
        //      cout << p << ", " << n_part[p] << endl;
        //        cout << p << "number of particles out of bounds " << nob[p] << endl;
    }
#pragma omp barrier
    //  cout << "get_density_checked out of bounds\n";

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

        // #pragma omp parallel for simd num_threads(nthreads) reduction (+: KEtot[0] ,nt[0],KEtot[1] ,nt[1] )
        for (int n = 0; n < n_part[p]; ++n)
        {
            KEtot[p] += v[p][0][n] * v[p][0][n] + v[p][1][n] * v[p][1][n] + v[p][2][n] * v[p][2][n];
            nt[p] += q[p][n];
        }
        KEtot[p] *= 0.5 * mp[p] / (e_charge_mass)*r_part_spart; // as if these particles were actually samples of the greater thing

#pragma omp barrier
#pragma omp parallel for simd num_threads(nthreads)
        for (int n = 0; n < n_part[p]; ++n)
        {
            v[p][0][n] *= (float)q[p][n];
            v[p][1][n] *= (float)q[p][n];
            v[p][2][n] *= (float)q[p][n];
        }
    }
#pragma omp barrier
#pragma omp parallel sections num_threads(nthreads)
    {
#pragma omp section
        {
            int p = 0;
            //   cout << "np :" << omp_get_thread_num() << endl;
            for (int n = 0; n < n_part[p]; ++n)
            {
                unsigned int i = ii[p][0][n], j = ii[p][1][n], k = ii[p][2][n];
                np[p][k][j][i] += (float)q[p][n];
                np_center[p][k][j][i][0] += (float)q[p][n] * offset[p][0][n];
                np_center[p][k][j][i][1] += (float)q[p][n] * offset[p][1][n];
                np_center[p][k][j][i][2] += (float)q[p][n] * offset[p][2][n];
            }
            smoothscalarfield(np[p], np_center[p], 0); // n
        }
#pragma omp section
        {
            int p = 1;
            //   cout << "np :" << omp_get_thread_num() << endl;
            for (int n = 0; n < n_part[p]; ++n)
            {
                unsigned int i = ii[p][0][n], j = ii[p][1][n], k = ii[p][2][n];
                np[p][k][j][i] += (float)q[p][n];
                np_center[p][k][j][i][0] += (float)q[p][n] * offset[p][0][n];
                np_center[p][k][j][i][1] += (float)q[p][n] * offset[p][1][n];
                np_center[p][k][j][i][2] += (float)q[p][n] * offset[p][2][n];
            }
            smoothscalarfield(np[p], np_center[p], 1); // p
        }
#pragma omp section
        {
            int p = 0;
            int c = 0;
            //          cout << "jx :" << p << " " << omp_get_thread_num() << endl;
            for (int n = 0; n < n_part[p]; ++n)
            {
                unsigned int i = ii[p][0][n], j = ii[p][1][n], k = ii[p][2][n];
                int s(v[p][c][n] > 0);
                jc2[p][s][c][k][j][i] += v[p][c][n];
                jc_center[p][s][c][k][j][i][0] += v[p][c][n] * offset[p][0][n];
                jc_center[p][s][c][k][j][i][1] += v[p][c][n] * offset[p][1][n];
                jc_center[p][s][c][k][j][i][2] += v[p][c][n] * offset[p][2][n];
            }
            smoothscalarfield(jc2[p][0][c], jc_center[p][0][c], 0); // jxp
            smoothscalarfield(jc2[p][1][c], jc_center[p][1][c], 1); // jxn
        }
#pragma omp section
        { //          cout << "jx :" << p << " " << omp_get_thread_num() << endl;
            int p = 1;
            int c = 0;
            for (int n = 0; n < n_part[p]; ++n)
            {
                unsigned int i = ii[p][0][n], j = ii[p][1][n], k = ii[p][2][n];
                int s(v[p][c][n] > 0);
                jc2[p][s][c][k][j][i] += v[p][c][n];
                jc_center[p][s][c][k][j][i][0] += v[p][c][n] * offset[p][0][n];
                jc_center[p][s][c][k][j][i][1] += v[p][c][n] * offset[p][1][n];
                jc_center[p][s][c][k][j][i][2] += v[p][c][n] * offset[p][2][n];
            }
            smoothscalarfield(jc2[p][0][c], jc_center[p][0][c], 0); // jxp
            smoothscalarfield(jc2[p][1][c], jc_center[p][1][c], 1); // jxn
        }
#pragma omp section
        { //      cout << "jy :" << omp_get_thread_num() << endl;
            int p = 0;
            int c = 1;
            for (int n = 0; n < n_part[p]; ++n)
            {
                unsigned int i = ii[p][0][n], j = ii[p][1][n], k = ii[p][2][n];
                int s(v[p][c][n] > 0);
                jc2[p][s][c][k][j][i] += v[p][c][n];
                jc_center[p][s][c][k][j][i][0] += v[p][c][n] * offset[p][0][n];
                jc_center[p][s][c][k][j][i][1] += v[p][c][n] * offset[p][1][n];
                jc_center[p][s][c][k][j][i][2] += v[p][c][n] * offset[p][2][n];
            }
            smoothscalarfield(jc2[p][0][c], jc_center[p][0][c], 0); // p
            smoothscalarfield(jc2[p][1][c], jc_center[p][1][c], 1); // n
        }
#pragma omp section
        { //    cout << "jy :" << omp_get_thread_num() << endl;
            int p = 1;
            int c = 1;
            for (int n = 0; n < n_part[p]; ++n)
            {
                unsigned int i = ii[p][0][n], j = ii[p][1][n], k = ii[p][2][n];
                int s(v[p][c][n] > 0);
                jc2[p][s][c][k][j][i] += v[p][c][n];
                jc_center[p][s][c][k][j][i][0] += v[p][c][n] * offset[p][0][n];
                jc_center[p][s][c][k][j][i][1] += v[p][c][n] * offset[p][1][n];
                jc_center[p][s][c][k][j][i][2] += v[p][c][n] * offset[p][2][n];
            }
            smoothscalarfield(jc2[p][0][c], jc_center[p][0][c], 0); // p
            smoothscalarfield(jc2[p][1][c], jc_center[p][1][c], 1); // n
        }

#pragma omp section
        { //    cout << "jz :" << omp_get_thread_num() << endl;
            int p = 0;
            int c = 2;
            for (int n = 0; n < n_part[p]; ++n)
            {
                unsigned int i = ii[p][0][n], j = ii[p][1][n], k = ii[p][2][n];
                int s(v[p][c][n] > 0);
                jc2[p][s][c][k][j][i] += v[p][c][n];
                jc_center[p][s][c][k][j][i][0] += v[p][c][n] * offset[p][0][n];
                jc_center[p][s][c][k][j][i][1] += v[p][c][n] * offset[p][1][n];
                jc_center[p][s][c][k][j][i][2] += v[p][c][n] * offset[p][2][n];
            }
            smoothscalarfield(jc2[p][0][c], jc_center[p][0][c], 0); // p
            smoothscalarfield(jc2[p][1][c], jc_center[p][1][c], 1); // n
        }

#pragma omp section
        { //   cout << "jz :" << omp_get_thread_num() << endl;
            int p = 1;
            int c = 2;
            for (int n = 0; n < n_part[p]; ++n)
            {
                unsigned int i = ii[p][0][n], j = ii[p][1][n], k = ii[p][2][n];
                int s(v[p][c][n] > 0);
                jc2[p][s][c][k][j][i] += v[p][c][n];
                jc_center[p][s][c][k][j][i][0] += v[p][c][n] * offset[p][0][n];
                jc_center[p][s][c][k][j][i][1] += v[p][c][n] * offset[p][1][n];
                jc_center[p][s][c][k][j][i][2] += v[p][c][n] * offset[p][2][n];
            }
            smoothscalarfield(jc2[p][0][c], jc_center[p][0][c], 0); // p
            smoothscalarfield(jc2[p][1][c], jc_center[p][1][c], 1); // n
        }
    }

#pragma omp barrier
    for (unsigned int i = 0; i < n_cells * 3; i++)
    {
        (reinterpret_cast<float *>(currentj[0]))[i] = (reinterpret_cast<float *>(jc2[0][0]))[i] + (reinterpret_cast<float *>(jc2[0][1]))[i];
        (reinterpret_cast<float *>(currentj[1]))[i] = (reinterpret_cast<float *>(jc2[1][0]))[i] + (reinterpret_cast<float *>(jc2[1][1]))[i];
    }
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
#pragma omp barrier
}
