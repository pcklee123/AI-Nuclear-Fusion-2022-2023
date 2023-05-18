#include "include/traj.h"

void get_densityfields(float currentj[2][3][n_space_divz][n_space_divy][n_space_divx],
                       float np[2][n_space_divz][n_space_divy][n_space_divx],
                       float npt[n_space_divz][n_space_divy][n_space_divx],
                       int nt[2], float KEtot[2], float posL[3], float posH[3], float dd[3],
                       float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd],
                       float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd],
                       int q[2][n_partd], float dt[2], int n_part[2],
                       float jc[3][n_space_divz][n_space_divy][n_space_divx])
{
    // find number of particle and current density fields
    // set fields=0 in preparation

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

    /*
    static float cx[2][n_parte];
    static float cy[2][n_parte];
    static float cz[2][n_parte];

    static unsigned int ii[2][n_parte];
    static unsigned int jj[2][n_parte];
    static unsigned int kk[2][n_parte];

    static float vx[2][n_parte];
    static float vy[2][n_parte];
    static float vz[2][n_parte];

    static float fracx[2][n_parte];
    static float fracy[2][n_parte];
    static float fracz[2][n_parte];
    static float fracx1[2][n_parte];
    static float fracy1[2][n_parte];
    static float fracz1[2][n_parte];

    static float total[2][n_parte];

    static float c0[2][n_parte];
    static float c1[2][n_parte];
    static float c2[2][n_parte];
    static float c3[2][n_parte];
    static float c4[2][n_parte];
    static float c5[2][n_parte];
    static float c6[2][n_parte];
    static float c7[2][n_parte];

    static float c0x[2][n_parte];
    static float c1x[2][n_parte];
    static float c2x[2][n_parte];
    static float c3x[2][n_parte];
    static float c4x[2][n_parte];
    static float c5x[2][n_parte];
    static float c6x[2][n_parte];
    static float c7x[2][n_parte];

    static float c0y[2][n_parte];
    static float c1y[2][n_parte];
    static float c2y[2][n_parte];
    static float c3y[2][n_parte];
    static float c4y[2][n_parte];
    static float c5y[2][n_parte];
    static float c6y[2][n_parte];
    static float c7y[2][n_parte];

    static float c0z[2][n_parte];
    static float c1z[2][n_parte];
    static float c2z[2][n_parte];
    static float c3z[2][n_parte];
    static float c4z[2][n_parte];
    static float c5z[2][n_parte];
    static float c6z[2][n_parte];
    static float c7z[2][n_parte];
    */
    static auto cx = new float[2][n_parte];
    static auto cy = new float[2][n_parte];
    static auto cz = new float[2][n_parte];

    static auto ii = new unsigned int[2][n_parte];
    static auto jj = new unsigned int[2][n_parte];
    static auto kk = new unsigned int[2][n_parte];
    static auto vc = new float[2][3][n_parte];
    //  static auto vx = new float[2][n_parte];
    //   static auto vy = new float[2][n_parte];
    //   static auto vz = new float[2][n_parte];

    static auto fracx = new float[2][n_parte];
    static auto fracy = new float[2][n_parte];
    static auto fracz = new float[2][n_parte];
    static auto fracx1 = new float[2][n_parte];
    static auto fracy1 = new float[2][n_parte];
    static auto fracz1 = new float[2][n_parte];

    static auto total = new float[2][n_parte];

    static auto c0 = new float[2][n_parte];
    static auto c1 = new float[2][n_parte];
    static auto c2 = new float[2][n_parte];
    static auto c3 = new float[2][n_parte];
    static auto c4 = new float[2][n_parte];
    static auto c5 = new float[2][n_parte];
    static auto c6 = new float[2][n_parte];
    static auto c7 = new float[2][n_parte];

    static auto c0x = new float[2][n_parte];
    static auto c1x = new float[2][n_parte];
    static auto c2x = new float[2][n_parte];
    static auto c3x = new float[2][n_parte];
    static auto c4x = new float[2][n_parte];
    static auto c5x = new float[2][n_parte];
    static auto c6x = new float[2][n_parte];
    static auto c7x = new float[2][n_parte];

    static auto c0y = new float[2][n_parte];
    static auto c1y = new float[2][n_parte];
    static auto c2y = new float[2][n_parte];
    static auto c3y = new float[2][n_parte];
    static auto c4y = new float[2][n_parte];
    static auto c5y = new float[2][n_parte];
    static auto c6y = new float[2][n_parte];
    static auto c7y = new float[2][n_parte];

    static auto c0z = new float[2][n_parte];
    static auto c1z = new float[2][n_parte];
    static auto c2z = new float[2][n_parte];
    static auto c3z = new float[2][n_parte];
    static auto c4z = new float[2][n_parte];
    static auto c5z = new float[2][n_parte];
    static auto c6z = new float[2][n_parte];
    static auto c7z = new float[2][n_parte];

#pragma omp parallel num_threads(2)
    {
        int p = omp_get_thread_num();
        //       cout << "p1=" << p << ",";
        for (int n = 0; n < n_part[p]; ++n)
        {
            // replace out of bounds particles and reduce the number of particles.
            if ((pos1x[p][n] <= xld) | (pos1y[p][n] <= yld) | (pos1z[p][n] <= zld) |
                (pos1x[p][n] >= xhd) | (pos1y[p][n] >= yhd) | (pos1z[p][n] >= zhd) |
                (pos0x[p][n] <= xld) | (pos0y[p][n] <= yld) | (pos0z[p][n] <= zld) |
                (pos0x[p][n] >= xhd) | (pos0y[p][n] >= yhd) | (pos0z[p][n] >= zhd))
            {
                int last = --n_part[p];
                //               cout << "n=" << n << ", npart[" << p << "]=" << n_part[p] << ":" << pos0x[p][n] << ", " << pos0y[p][n] << ", " << pos0z[p][n] << endl;
                pos0x[p][n] = pos0x[p][last];
                pos0y[p][n] = pos0y[p][last];
                pos0z[p][n] = pos0z[p][last];
                pos1x[p][n] = pos1x[p][last];
                pos1y[p][n] = pos1y[p][last];
                pos1z[p][n] = pos1z[p][last];
                q[p][n] = q[p][last];
                q[p][last] = 0;
                //                cout << "n=" << n << ", npart[" << p << "]=" << n_part[p] << ":" << pos1x[p][n] << ", " << pos1y[p][n] << ", " << pos1z[p][n] << endl;
                n--;
            }
        }
    }
#pragma omp barrier
    //        cout << "get_densitya\n";
    float dxi = 1.0 / dd[0];
    float dyi = 1.0 / dd[1];
    float dzi = 1.0 / dd[2];
    float ds2 = (dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2]) / 1e8;
#pragma omp parallel num_threads(2)
    {
        int p = omp_get_thread_num();
        //                cout << "p2=" << p << ",";
        //                   float charge = q[p][n];
#pragma omp parallel for simd num_threads(nthreads)
        for (int n = 0; n < n_part[p]; ++n)
        {
            cx[p][n] = (pos1x[p][n] - posL[0]) * dxi;
            cy[p][n] = (pos1y[p][n] - posL[1]) * dyi;
            cz[p][n] = (pos1z[p][n] - posL[2]) * dzi;

            ii[p][n] = cx[p][n];
            jj[p][n] = cy[p][n];
            kk[p][n] = cz[p][n];

            fracx[p][n] = cx[p][n] - ii[p][n];
            fracy[p][n] = cy[p][n] - jj[p][n];
            fracz[p][n] = cz[p][n] - kk[p][n];

            fracx1[p][n] = 1.0 - fracx[p][n];
            fracy1[p][n] = 1.0 - fracy[p][n];
            fracz1[p][n] = 1.0 - fracz[p][n];

            // I'm not sure whether this algorithm is correct, but it seems legit
            // Only drawback is that it is not exact (a tiny fraction of charge leaks)
            // float fx_2 = fracx[p][n] * fracx[p][n], fy_2 = fracy[p][n] * fracy[p][n], fz_2 = fracz[p][n] * fracz[p][n];
            // float fx1_2 = fracx1[p][n] * fracx1[p][n], fy1_2 = fracy1[p][n] * fracy1[p][n], fz1_2 = fracz1[p][n] * fracz1[p][n];
            c0x[p][n] = fracx[p][n] * fracx[p][n];
            c0y[p][n] = fracy[p][n] * fracy[p][n];
            c0z[p][n] = fracz[p][n] * fracz[p][n];
            c1x[p][n] = fracx1[p][n] * fracx1[p][n];
            c1y[p][n] = fracy1[p][n] * fracy1[p][n];
            c1z[p][n] = fracz1[p][n] * fracz1[p][n];

            // The problem with any distribution is that it will end up pushing itself, although overall it still leads to slightly better stability
            c0[p][n] = 1.f / (c0x[p][n] + c0y[p][n] + c0z[p][n] + ds2);
            c1[p][n] = 1.f / (c1x[p][n] + c0y[p][n] + c0z[p][n] + ds2);
            c2[p][n] = 1.f / (c0x[p][n] + c1y[p][n] + c0z[p][n] + ds2);
            c3[p][n] = 1.f / (c1x[p][n] + c1y[p][n] + c0z[p][n] + ds2);
            c4[p][n] = 1.f / (c0x[p][n] + c0y[p][n] + c1z[p][n] + ds2);
            c5[p][n] = 1.f / (c1x[p][n] + c0y[p][n] + c1z[p][n] + ds2);
            c6[p][n] = 1.f / (c0x[p][n] + c1y[p][n] + c1z[p][n] + ds2);
            c7[p][n] = 1.f / (c1x[p][n] + c1y[p][n] + c1z[p][n] + ds2);

            total[p][n] = (float)q[p][n] / (c0[p][n] + c1[p][n] + c2[p][n] + c3[p][n] + c4[p][n] + c5[p][n] + c6[p][n] + c7[p][n]); // Multiply r... by charge, ie total /= charge.
                                                                                                                                    // Then take the reciprocal for multiplication (faster)
            c0[p][n] *= total[p][n];
            c1[p][n] *= total[p][n];
            c2[p][n] *= total[p][n];
            c3[p][n] *= total[p][n];
            c4[p][n] *= total[p][n];
            c5[p][n] *= total[p][n];
            c6[p][n] *= total[p][n];
            c7[p][n] *= total[p][n];

            vc[p][0][n] = (pos1x[p][n] - pos0x[p][n]) / dt[p];
            vc[p][1][n] = (pos1y[p][n] - pos0y[p][n]) / dt[p];
            vc[p][2][n] = (pos1z[p][n] - pos0z[p][n]) / dt[p];

            c0x[p][n] = c0[p][n] * vc[p][0][n];
            c1x[p][n] = c1[p][n] * vc[p][0][n];
            c2x[p][n] = c2[p][n] * vc[p][0][n];
            c3x[p][n] = c3[p][n] * vc[p][0][n];
            c4x[p][n] = c4[p][n] * vc[p][0][n];
            c5x[p][n] = c5[p][n] * vc[p][0][n];
            c6x[p][n] = c6[p][n] * vc[p][0][n];
            c7x[p][n] = c7[p][n] * vc[p][0][n];

            c0y[p][n] = c0[p][n] * vc[p][1][n];
            c1y[p][n] = c1[p][n] * vc[p][1][n];
            c2y[p][n] = c2[p][n] * vc[p][1][n];
            c3y[p][n] = c3[p][n] * vc[p][1][n];
            c4y[p][n] = c4[p][n] * vc[p][1][n];
            c5y[p][n] = c5[p][n] * vc[p][1][n];
            c6y[p][n] = c6[p][n] * vc[p][1][n];
            c7y[p][n] = c7[p][n] * vc[p][1][n];

            c0z[p][n] = c0[p][n] * vc[p][2][n];
            c1z[p][n] = c1[p][n] * vc[p][2][n];
            c2z[p][n] = c2[p][n] * vc[p][2][n];
            c3z[p][n] = c3[p][n] * vc[p][2][n];
            c4z[p][n] = c4[p][n] * vc[p][2][n];
            c5z[p][n] = c5[p][n] * vc[p][2][n];
            c6z[p][n] = c6[p][n] * vc[p][2][n];
            c7z[p][n] = c7[p][n] * vc[p][2][n];
        }
#pragma omp parallel for simd num_threads(8)
        for (int n = 0; n < n_part[p]; ++n)
        {
            KEtot[p] += vc[p][0][n] * vc[p][0][n] + vc[p][1][n] * vc[p][1][n] + vc[p][2][n] * vc[p][2][n];
        }
        KEtot[p] *= 0.5 * mp[p] / e_charge_mass;
        KEtot[p] *= r_part_spart; // as if these particles were actually samples of the greater thing
                                  //        cout << "get_density\n";        //      cout <<maxk <<",";
        for (int n = 0; n < n_part[p]; ++n)
        {
            unsigned int i, j, k;
            i = ii[p][n];
            j = jj[p][n];
            k = kk[p][n];

            // number of charge (in units of 1.6e-19 C) in each cell
            np[p][k][j][i] += c0[p][n];
            np[p][k][j][i + 1] += c1[p][n];
            np[p][k][j + 1][i] += c2[p][n];
            np[p][k][j + 1][i + 1] += c3[p][n];
            np[p][k + 1][j][i] += c4[p][n];
            np[p][k + 1][j][i + 1] += c5[p][n];
            np[p][k + 1][j + 1][i] += c6[p][n];
            np[p][k + 1][j + 1][i + 1] += c7[p][n];
            nt[p] += q[p][n];
            // cout << coeffs[0] + coeffs[1] + coeffs[2] + coeffs[3] + coeffs[4] + coeffs[5] + coeffs[6] + coeffs[7] << endl;
            //  current density p=0 electron j=nev in each cell n in units 1.6e-19 C m/s

            currentj[p][0][k][j][i] += c0x[p][n];
            currentj[p][0][k][j][i + 1] += c1x[p][n];
            currentj[p][0][k][j + 1][i] += c2x[p][n];
            currentj[p][0][k][j + 1][i + 1] += c3x[p][n];
            currentj[p][0][k + 1][j][i] += c4x[p][n];
            currentj[p][0][k + 1][j][i + 1] += c5x[p][n];
            currentj[p][0][k + 1][j + 1][i] += c6x[p][n];
            currentj[p][0][k + 1][j + 1][i + 1] += c7x[p][n];

            currentj[p][1][k][j][i] += c0y[p][n];
            currentj[p][1][k][j][i + 1] += c1y[p][n];
            currentj[p][1][k][j + 1][i] += c2y[p][n];
            currentj[p][1][k][j + 1][i + 1] += c3y[p][n];
            currentj[p][1][k + 1][j][i] += c4y[p][n];
            currentj[p][1][k + 1][j][i + 1] += c5y[p][n];
            currentj[p][1][k + 1][j + 1][i] += c6y[p][n];
            currentj[p][1][k + 1][j + 1][i + 1] += c7y[p][n];

            currentj[p][2][k][j][i] += c0z[p][n];
            currentj[p][2][k][j][i + 1] += c1z[p][n];
            currentj[p][2][k][j + 1][i] += c2z[p][n];
            currentj[p][2][k][j + 1][i + 1] += c3z[p][n];
            currentj[p][2][k + 1][j][i] += c4z[p][n];
            currentj[p][2][k + 1][j][i + 1] += c5z[p][n];
            currentj[p][2][k + 1][j + 1][i] += c6z[p][n];
            currentj[p][2][k + 1][j + 1][i + 1] += c7z[p][n];
        }
    }

#pragma omp barrier

#pragma omp parallel for simd
    for (unsigned int i = 0; i < n_cells * 3; i++)
    {
        (reinterpret_cast<float *>(jc))[i] = (reinterpret_cast<float *>(currentj[0]))[i] + (reinterpret_cast<float *>(currentj[1]))[i];
    }
#pragma omp parallel for simd
    for (unsigned int i = 0; i < n_cells; i++)
    {
        (reinterpret_cast<float *>(npt))[i] = (reinterpret_cast<float *>(np[0]))[i] + (reinterpret_cast<float *>(np[1]))[i];
    }
    //       cout << "get_density-done\n";
    //    delete[] cx, cy, cz, ii, jj, kk, vx, vy, vz, fracx, fracy, fracz, total, c0, c1, c2, c3, c4, c5, c6, c7, c0x, c1x, c2x, c3x, c4x, c5x, c6x, c7x, c0y, c1y, c2y, c3y, c4y, c5y, c6y, c7y, c0z, c1z, c2z, c3z, c4z, c5z, c6z, c7z;
}

/*
void get_densityfields(float currentj[2][3][n_space_divz][n_space_divy][n_space_divx],
                       float np[2][n_space_divz][n_space_divy][n_space_divx],
                       float npt[n_space_divz][n_space_divy][n_space_divx],
                       int nt[2], float KEtot[2], float posL[3], float posH[3], float dd[3],
                       float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd],
                       float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd],
                       int q[2][n_partd], float dt[2], int n_part[2],
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

    for (int p = 0; p < 2; p++)
    {
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

    for (int p = 0; p < 2; p++)
    {
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
            //      cout << "npp" << p << n_part[p] << endl;
            // #pragma omp distribute parallel for reduction(+ \
                                              : np[p], nt[p], currentj[p], KEtot[p])
            for (int n = start_index; n < end_index; n++)
            {
                unsigned int i, j, k, c;
                i = (unsigned int)((pos1x[p][n] - posL[0]) / dd[0]);
                j = (unsigned int)((pos1y[p][n] - posL[1]) / dd[1]);
                k = (unsigned int)((pos1z[p][n] - posL[2]) / dd[2]);
#pragma omp atomic
                np[p][k][j][i] += q[p][n];
#pragma omp atomic
                nt[p] += q[p][n];
                float dx = pos1x[p][n] - pos0x[p][n], dy = pos1y[p][n] - pos0y[p][n], dz = pos1z[p][n] - pos0z[p][n];
                float vx = dx / dt[p], vy = dy / dt[p], vz = dz / dt[p];
#pragma omp atomic
                currentj[p][0][k][j][i] += q[p][n] * vx;
#pragma omp atomic
                currentj[p][1][k][j][i] += q[p][n] * vy;
#pragma omp atomic
                currentj[p][2][k][j][i] += q[p][n] * vz;
#pragma omp atomic
                KEtot[p] += dx * dx + dy * dy + dz * dz;
            }
            KEtot[p] *= 0.5 * mp[p] / (e_charge_mass * dt[p] * dt[p])*r_part_spart; // as if these particles were actually samples of the greater thing
            //      cout <<maxk <<",";
        }
    }
#pragma omp barrier
//  cout << "get_density";
#pragma omp parallel for simd
    for (unsigned int i = 0; i < n_cells * 3; i++)
    {
        (reinterpret_cast<float *>(jc))[i] = (reinterpret_cast<float *>(currentj[0]))[i] + (reinterpret_cast<float *>(currentj[1]))[i];
    }
#pragma omp parallel for simd
    for (unsigned int i = 0; i < n_cells; i++)
    {
        (reinterpret_cast<float *>(npt))[i] = (reinterpret_cast<float *>(np[0]))[i] + (reinterpret_cast<float *>(np[1]))[i];
    }
}
#ifdef LinearDistribution // Distribute the particle according to 1/r, not defining this will distribute according to 1/r2
            float coeffs[8] = {
                fracx1 * fracy1 * fracz1,
                fracx * fracy1 * fracz1,
                fracx1 * fracy * fracz1,
                fracx * fracy * fracz1,
                fracx1 * fracy1 * fracz,
                fracx * fracy1 * fracz,
                fracx1 * fracy * fracz,
                fracx * fracy * fracz};
#pragma unroll(8)
            for (int i = 0; i < 8; ++i)
                coeffs[i] *= charge;
#else
*/