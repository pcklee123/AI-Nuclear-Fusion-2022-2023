#include "include/traj.h"
// Interpolate the value at a given point

void get_densityfields(fields *fi, particles *pt, par *par)
{
    return;
    // find number of particle and current density fields
    // set limits beyond which particle is considered as "lost"
    static const float ddi[3] = {1.f / par->dd[0], 1.f / par->dd[1], 1.f / par->dd[2]}; // precalculate reciprocals
    static const float dti[2] = {1.f / par->dt[0], 1.f / par->dt[1]};
    // cout << " par->dt[0]" << par->dt[0] << endl;
    // cell indices for each particle [2][3][n_parte]
    static auto *ii = static_cast<unsigned int(*)[3][n_parte]>(_aligned_malloc(2 * 3 * n_parte * sizeof(unsigned int), alignment));
    // particle velocity array [2][3][n_parte]
    static auto *v = static_cast<float(*)[3][n_parte]>(_aligned_malloc(2 * 3 * n_parte * sizeof(float), alignment));
    // particle offsets array [2][3][n_parte]
    static auto *offset = static_cast<float(*)[3][n_parte]>(_aligned_malloc(2 * 3 * n_parte * sizeof(float), alignment));
    // temp space 7 1. density e + i 2-7 current x,y,z p,n
    static auto *ftemp = static_cast<float(*)[n_space_divz][n_space_divy][n_space_divx]>(_aligned_malloc(7 * n_cells * sizeof(float), alignment));
    // center of charge field arrays [2-particle type][3 pos][z][y][x]
    static auto *np_center = static_cast<float(*)[n_space_divz][n_space_divy][n_space_divx][3]>(_aligned_malloc(2 * 3 * n_cells * sizeof(float), alignment));
    // center of current field arrays [2][3-pos][3-current component][z][y][x]
    static auto *jc2 = static_cast<float(*)[2][3][n_space_divz][n_space_divy][n_space_divx]>(_aligned_malloc(2 * 2 * 3 * n_cells * sizeof(float), alignment));
    static auto *jc_center = static_cast<float(*)[2][3][n_space_divz][n_space_divy][n_space_divx][3]>(_aligned_malloc(2 * 2 * 3 * 3 * n_cells * sizeof(float), alignment));

    // set fields=0 in preparation// Could split into threads
  //  fill(reinterpret_cast<float *>(fi->np), reinterpret_cast<float *>(fi->np) + n_cells, 0.f);
    //   fill(reinterpret_cast<float *>(currentj), reinterpret_cast<float *>(currentj) + n_cells * 2 * 3, 0.f);
    fill(reinterpret_cast<float *>(ftemp), reinterpret_cast<float *>(ftemp) + n_cells * 7, 0.f);
    fill(reinterpret_cast<float *>(np_center), reinterpret_cast<float *>(np_center) + n_cells * 3 * 2, 0.f);
    fill(reinterpret_cast<float *>(jc2), reinterpret_cast<float *>(jc2) + n_cells * 2 * 2 * 3, 0.f);
    fill(reinterpret_cast<float *>(jc_center), reinterpret_cast<float *>(jc_center) + n_cells * 2 * 2 * 3 * 3, 0.f);

    static auto oblist = new unsigned int[2][n_parte]; // list of out of bound particles
    static auto iblist = new unsigned int[2][n_parte];
    int nob[2]; // number of particles out of bounds
    int nib[2]; // number of particles within bounds

    //  cout << "get_density_start\n";
// remove out of bounds points and get x,y,z index of each particle
#pragma omp parallel for num_threads(2)
    for (int p = 0; p < 2; ++p)
    {
#pragma omp parallel sections
        {
#pragma omp section
#pragma omp parallel for simd
            for (unsigned int n = 0; n < par->n_part[p]; ++n)
            {
                bool toolow = pt->pos1x[p][n] <= par->posL_1[0], toohigh = pt->pos1x[p][n] >= par->posH_1[0];
                pt->pos1x[p][n] = toolow ? par->posL_15[0] : (toohigh ? par->posH_15[0] : pt->pos1x[p][n]);
                pt->pos0x[p][n] = (toolow || toohigh) ? pt->pos1x[p][n] : pt->pos0x[p][n];
                pt->q[p][n] = (toolow || toohigh) ? 0 : pt->q[p][n];
            }
#pragma omp section
#pragma omp parallel for simd
            for (unsigned int n = 0; n < par->n_part[p]; ++n)
            {
                bool toolow = pt->pos1y[p][n] <= par->posL_1[1], toohigh = pt->pos1y[p][n] >= par->posH_1[1];
                pt->pos1y[p][n] = toolow ? par->posL_15[1] : (toohigh ? par->posH_15[1] : pt->pos1y[p][n]);
                pt->pos0y[p][n] = (toolow || toohigh) ? pt->pos1y[p][n] : pt->pos0y[p][n];
                pt->q[p][n] = (toolow || toohigh) ? 0 : pt->q[p][n];
            }
#pragma omp section
            for (unsigned int n = 0; n < par->n_part[p]; ++n)
            {
#ifdef sphere
                bool toolow = pt->pos1z[p][n] <= par->posL_1[2], toohigh = pt->pos1z[p][n] >= par->posH_1[2];
                pt->pos1z[p][n] = toolow ? par->posL_15[2] : (toohigh ? par->posH_15[2] : pt->pos1z[p][n]);
                pt->pos0z[p][n] = (toolow || toohigh) ? pt->pos1z[p][n] : pt->pos0z[p][n];
                pt->q[p][n] = (toolow || toohigh) ? 0 : pt->q[p][n];
#endif
#ifdef cylinder // rollover particles in z direction
                bool toolow = pt->pos1z[p][n] <= par->posL_1[2], toohigh = pt->pos1z[p][n] >= par->posH_1[2];
                pt->pos1z[p][n] += toolow ? (n_space_divz - 2) * par->dd[2] : (toohigh ? (2 - n_space_divz) * par->dd[2] : 0);
                pt->pos0z[p][n] += toolow ? (n_space_divz - 2) * par->dd[2] : (toohigh ? (2 - n_space_divz) * par->dd[2] : 0);
#endif
            }
        }
    }
#pragma omp barrier

#pragma omp parallel for num_threads(2)
    for (int p = 0; p < 2; ++p)
    {
#pragma omp parallel for simd
        for (unsigned int n = 0; n < par->n_part[p]; ++n) // get cell indices (x,y,z) a particle belongs to
        {
            ii[p][0][n] = (int)roundf((pt->pos1x[p][n] - par->posL[0]) * ddi[0]);
            offset[p][0][n] = (pt->pos1x[p][n] - par->posL[0]) * ddi[0] - (float)(ii[p][0][n]);
            ii[p][1][n] = (int)roundf((pt->pos1y[p][n] - par->posL[1]) * ddi[1]);
            offset[p][1][n] = (pt->pos1y[p][n] - par->posL[1]) * ddi[1] - (float)(ii[p][1][n]);
            ii[p][2][n] = (int)roundf((pt->pos1z[p][n] - par->posL[2]) * ddi[2]);
            offset[p][2][n] = (pt->pos1z[p][n] - par->posL[2]) * ddi[2] - (float)(ii[p][2][n]);
        }
    }

#pragma omp barrier
    //  cout << "get_density_checked out of bounds\n";

    // #pragma omp parallel for num_threads(2)
    for (int p = 0; p < 2; ++p)
    {
        // #pragma omp parallel for simd
        for (int n = 0; n < par->n_part[p]; ++n)
        {
            v[p][0][n] = pt->q[p][n] * (pt->pos1x[p][n] - pt->pos0x[p][n]) * dti[p];
            v[p][1][n] = pt->q[p][n] * (pt->pos1y[p][n] - pt->pos0y[p][n]) * dti[p];
            v[p][2][n] = pt->q[p][n] * (pt->pos1z[p][n] - pt->pos0z[p][n]) * dti[p];
        }
    }

#pragma omp barrier

#pragma omp parallel sections
    {
#pragma omp section
        {
#pragma omp parallel for num_threads(2)
            for (int p = 0; p < 2; ++p)
            {
                for (int n = 0; n < par->n_part[p]; ++n)
                {
                    unsigned int i = ii[p][0][n], j = ii[p][1][n], k = ii[p][2][n];
                    //fi->np[p][k][j][i] += (pt->q[p][n]);
                    np_center[p][k][j][i][0] += (pt->q[p][n]) * offset[p][0][n];
                    np_center[p][k][j][i][1] += (pt->q[p][n]) * offset[p][1][n];
                    np_center[p][k][j][i][2] += (pt->q[p][n]) * offset[p][2][n];
                }
            }
            
            smoothscalarfield(fi->np[0], ftemp[0], np_center[0], 0);
            memcpy(reinterpret_cast<float *>(fi->np[0]), reinterpret_cast<float *>(ftemp[0]), n_cells * sizeof(float));
            smoothscalarfield(fi->np[1], ftemp[0], np_center[1], 1);
            // npt is smoothed np[0] and np[1] are not
            memcpy(reinterpret_cast<float *>(fi->npt), reinterpret_cast<float *>(ftemp[0]), n_cells * sizeof(float));
            
            /*
#pragma omp parallel for simd num_threads(nthreads)
            for (unsigned int i = 0; i < n_cells; i++)
                (reinterpret_cast<float *>(fi->npt))[i] = (reinterpret_cast<float *>(fi->np[0]))[i] + (reinterpret_cast<float *>(fi->np[1]))[i];
            //*/
        }
#pragma omp section
        {
#pragma omp parallel for num_threads(6)
            for (int pc = 0; pc < 6; ++pc)
            {
                int p = pc / 3, c = pc % 3;
                int pc2 = pc + 1;
                int te = (pc + 1) * 2;
                for (int n = 0; n < par->n_part[p]; ++n)
                {
                    unsigned int i = ii[p][0][n], j = ii[p][1][n], k = ii[p][2][n];
                    int s = static_cast<int>(v[p][c][n] > 0);
                    jc2[p][s][c][k][j][i] += v[p][c][n];
                    jc_center[p][s][c][k][j][i][0] += v[p][c][n] * offset[p][0][n];
                    jc_center[p][s][c][k][j][i][1] += v[p][c][n] * offset[p][1][n];
                    jc_center[p][s][c][k][j][i][2] += v[p][c][n] * offset[p][2][n];
                }
                smoothscalarfield(jc2[p][0][c], ftemp[pc2], jc_center[p][0][c], te);
                smoothscalarfield(jc2[p][1][c], ftemp[pc2], jc_center[p][1][c], te + 1);
                memcpy(reinterpret_cast<float *>(fi->currentj[p][c]), reinterpret_cast<float *>(ftemp[pc2]), n_cells * sizeof(float));
            }
        }
    }
#pragma omp barrier

// cout << "Max Np0" << maxvalf(reinterpret_cast<float *>(np[0]), n_cells) << endl;
// cout << "Max Np1" << maxvalf(reinterpret_cast<float *>(np[1]), n_cells) << endl;
// cout << "Max Npt" << maxvalf(reinterpret_cast<float *>(npt), n_cells) << endl;
#pragma omp parallel for simd num_threads(nthreads)
    for (unsigned int i = 0; i < n_cells * 3; i++)
        (reinterpret_cast<float *>(fi->jc))[i] = (reinterpret_cast<float *>(fi->currentj[0]))[i] + (reinterpret_cast<float *>(fi->currentj[1]))[i];
#pragma omp barrier
}
