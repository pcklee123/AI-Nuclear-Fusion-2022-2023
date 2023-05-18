#include "include/traj.h"
#define VEC_LEN 8 // Define the vector length for AVX256
// Interpolate the value at a given point
void smoothscalarfield(float f[n_space_divz][n_space_divy][n_space_divx], float ftemp[n_space_divz][n_space_divy][n_space_divx],
                       float fc[n_space_divz][n_space_divy][n_space_divx][3], int s)
{
    float d;
    if ((s % 2) == 1)
        d = 0.00001f;
    else
        d = -0.00001f;
    // cout << "smoothfield" << endl;
    // calculate center of charge field as offsets (-0.5 to 0.5) from cell center
    // cout << "calculate center of charge field" << endl;

    int i, j, k;
    for (k = 0; k < n_space_divz; ++k)
    {
        for (j = 0; j < n_space_divy; ++j)
        {
            // Load the denominator into a vector register
            __m256 denom = _mm256_set1_ps(d);

            for (i = 0; i < n_space_divx; i += VEC_LEN)
            {
                // Load four sets of 3D vector values into four vector registers
                __m256 fc0 = _mm256_load_ps(&fc[k][j][i][0]);
                __m256 fc1 = _mm256_load_ps(&fc[k][j][i + 4][0]);
                __m256 f0 = _mm256_load_ps(&f[k][j][i]);
                __m256 f1 = _mm256_load_ps(&f[k][j][i + 4]);

                // Add the denominator to the f vector
                f0 = _mm256_add_ps(f0, denom);
                f1 = _mm256_add_ps(f1, denom);

                // Divide the four sets of 3D vectors by their corresponding f values
                fc0 = _mm256_div_ps(fc0, f0);
                fc1 = _mm256_div_ps(fc1, f1);

                // Store the four sets of 3D vectors back into the fc array
                _mm256_store_ps(&fc[k][j][i][0], fc0);
                _mm256_store_ps(&fc[k][j][i + 4][0], fc1);
            }
        }
    }

    // calculate the 8 coefficients out of 27 and their indices
    /* center is [0][0][0] [dk][dj][di]so [-1][-1][-1],[-1][-1][0],[-1][-1][1] ... dk*n_space_divx*n_space_divx+dj*n_space_divx+di */
    int k1, j1, i1, sw;
    float fx0, fx1, fy0, fy1, fz0, fz1;
//   cout << "smoothfield calculate" << endl;
#pragma omp parallel for simd
    for (int k0 = 1; k0 < (n_space_divz - 1); ++k0)
        for (int j0 = 1; j0 < (n_space_divy - 1); ++j0)

            for (int i0 = 1; i0 < (n_space_divx - 1); ++i0)
            {
                sw = (fc[k0][j0][i0][2] > 0.0f ? 4 : 0) + (fc[k0][j0][i0][1] > 0.0f ? 2 : 0) + (fc[k0][j0][i0][0] > 0.0f ? 1 : 0);
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
}
