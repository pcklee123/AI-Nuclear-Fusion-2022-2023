#include "include/traj.h"
// Interpolate the value at a given point
void smoothscalarfield(float f[n_space_divz][n_space_divy][n_space_divx],
                       float fc[n_space_divz][n_space_divy][n_space_divx][3], int s)
{
    float d;
    if (s == 1)
        d = 0.00001f;
    else
        d = -0.00001f;
    // cout << "smoothfield" << endl;
    auto *ftemp = static_cast<float(*)[n_space_divy][n_space_divx]>(_aligned_malloc(n_space_divz * n_space_divy * n_space_divx * sizeof(float), alignment));
    fill(reinterpret_cast<float *>(ftemp), reinterpret_cast<float *>(ftemp) + n_cells, 0.f);
    // calculate center of charge field as offsets (-0.5 to 0.5) from cell center
    //   cout << "calculate center of charge field" << endl;
    for (int k = 0; k < n_space_divz; ++k)
        for (int j = 0; j < n_space_divy; ++j)
            for (int i = 0; i < n_space_divx; ++i)
            {
                fc[k][j][i][0] = (fc[k][j][i][0] / (f[k][j][i] + d));
                fc[k][j][i][1] = (fc[k][j][i][1] / (f[k][j][i] + d));
                fc[k][j][i][2] = (fc[k][j][i][2] / (f[k][j][i] + d));
            }
            //why are there a bunch of stuff here?
    fc[1][1][1][0] = 0;
    fc[1][1][1][1] = 0;
    fc[1][1][1][2] = 0;
    // calculate the 8 coefficients out of 27 and their indices
    /* center is [0][0][0] [dk][dj][di]so [-1][-1][-1],[-1][-1][0],[-1][-1][1] ... dk*n_space_divx*n_space_divx+dj*n_space_divx+di */
    int k1, j1, i1, sw;
    float fx0, fx1, fy0, fy1, fz0, fz1;
    //   cout << "smoothfield calculate" << endl;
    for (int k0 = 1; k0 < (n_space_divz - 1); ++k0)
        for (int j0 = 1; j0 < (n_space_divy - 1); ++j0)
            for (int i0 = 1; i0 < (n_space_divx - 1); ++i0)
            {
                sw = ((int)(fc[k0][j0][i0][2] > 0.0f) << 2) + ((int)(fc[k0][j0][i0][1] > 0.0f) << 1) + (int)(fc[k0][j0][i0][0] > 0.0f);
                //              cout << (fc[k0][j0][i0][2] < 0.0f) << (fc[k0][j0][i0][1] < 0.0f) << (fc[k0][j0][i0][0] < 0.0f) << endl;
                switch (sw)
                {
                case 0: // 000 zyx
                    k1 = k0 - 1;
                    j1 = j0 - 1;
                    i1 = i0 - 1;
                    fz0 = -fc[k0][j0][i0][2];
                    fz1 = 1 + fc[k0][j0][i0][2];
                    fy0 = -fc[k0][j0][i0][1];
                    fy1 = 1 + fc[k0][j0][i0][1];
                    fx0 = -fc[k0][j0][i0][0];
                    fx1 = 1 + fc[k0][j0][i0][0];
                    break;
                case 1: // 001
                    k1 = k0 - 1;
                    j1 = j0 - 1;
                    i1 = i0 + 1;
                    fz0 = -fc[k0][j0][i0][2];
                    fz1 = 1 + fc[k0][j0][i0][2];
                    fy0 = -fc[k0][j0][i0][1];
                    fy1 = 1 + fc[k0][j0][i0][1];
                    fx0 = fc[k0][j0][i0][0];
                    fx1 = 1 - fc[k0][j0][i0][0];
                    break;
                case 2: // 010
                    k1 = k0 - 1;
                    j1 = j0 + 1;
                    i1 = i0 - 1;
                    fz0 = -fc[k0][j0][i0][2];
                    fz1 = 1 + fc[k0][j0][i0][2];
                    fy0 = fc[k0][j0][i0][1];
                    fy1 = 1 - fc[k0][j0][i0][1];
                    fx0 = -fc[k0][j0][i0][0];
                    fx1 = 1 + fc[k0][j0][i0][0];
                    break;
                case 3: // 011
                    k1 = k0 - 1;
                    j1 = j0 + 1;
                    i1 = i0 + 1;
                    fz0 = -fc[k0][j0][i0][2];
                    fz1 = 1 + fc[k0][j0][i0][2];
                    fy0 = fc[k0][j0][i0][1];
                    fy1 = 1 - fc[k0][j0][i0][1];
                    fx0 = fc[k0][j0][i0][0];
                    fx1 = 1 - fc[k0][j0][i0][0];
                    break;
                case 4: // 100
                    k1 = k0 + 1;
                    j1 = j0 - 1;
                    i1 = i0 - 1;
                    fz0 = fc[k0][j0][i0][2];
                    fz1 = 1 - fc[k0][j0][i0][2];
                    fy0 = -fc[k0][j0][i0][1];
                    fy1 = 1 + fc[k0][j0][i0][1];
                    fx0 = -fc[k0][j0][i0][0];
                    fx1 = 1 + fc[k0][j0][i0][0];
                    break;
                    break;
                case 5: // 101
                    k1 = k0 + 1;
                    j1 = j0 - 1;
                    i1 = i0 + 1;
                    fz0 = fc[k0][j0][i0][2];
                    fz1 = 1 - fc[k0][j0][i0][2];
                    fy0 = -fc[k0][j0][i0][1];
                    fy1 = 1 + fc[k0][j0][i0][1];
                    fx0 = fc[k0][j0][i0][0];
                    fx1 = 1 - fc[k0][j0][i0][0];
                    break;
                case 6: // 110
                    k1 = k0 + 1;
                    j1 = j0 + 1;
                    i1 = i0 - 1;
                    fz0 = fc[k0][j0][i0][2];
                    fz1 = 1 - fc[k0][j0][i0][2];
                    fy0 = fc[k0][j0][i0][1];
                    fy1 = 1 - fc[k0][j0][i0][1];
                    fx0 = -fc[k0][j0][i0][0];
                    fx1 = 1 + fc[k0][j0][i0][0];
                    break;
                case 7: // 111
                    k1 = k0 + 1;
                    j1 = j0 + 1;
                    i1 = i0 + 1;
                    fz0 = fc[k0][j0][i0][2];
                    fz1 = 1 - fc[k0][j0][i0][2];
                    fy0 = fc[k0][j0][i0][1];
                    fy1 = 1 - fc[k0][j0][i0][1];
                    fx0 = fc[k0][j0][i0][0];
                    fx1 = 1 - fc[k0][j0][i0][0];
                    break;
                default:
                    cout << "error smoothfield default " << sw << endl;
                }
                ftemp[k0][j0][i0] += f[k0][j0][i0] * fz1 * fy1 * fx1;
                ftemp[k1][j0][i0] += f[k0][j0][i0] * fz0 * fy1 * fx1;
                ftemp[k0][j1][i0] += f[k0][j0][i0] * fz1 * fy0 * fx1;
                ftemp[k1][j1][i0] += f[k0][j0][i0] * fz0 * fy0 * fx1;
                ftemp[k0][j0][i1] += f[k0][j0][i0] * fz1 * fy1 * fx0;
                ftemp[k1][j0][i1] += f[k0][j0][i0] * fz0 * fy1 * fx0;
                ftemp[k0][j1][i1] += f[k0][j0][i0] * fz1 * fy0 * fx0;
                ftemp[k1][j1][i1] += f[k0][j0][i0] * fz0 * fy0 * fx0;
            }
    //  cout << "smoothfield copy back" << endl;

 // #pragma omp parallel for
    for (int n = 0; n < n_cells ; ++n)
    {
     reinterpret_cast<float *>(f)[n] = reinterpret_cast<float *>(ftemp)[n];
    }
    /*
    for (int k = 0; k < n_space_divz; ++k)
    {
        for (int j = 0; j < n_space_divy; ++j)
        {
            for (int i = 0; i < n_space_divx; ++i)
            {
                f[k][j][i] = ftemp[k][j][i];
            }
        }
    }
    */
    _aligned_free(ftemp);
}
