#include "include/traj.h"
// Interpolate the value at a given point
void smoothscalarfield(float f[n_space_divz][n_space_divy][n_space_divx],
                       float fc[n_space_divz][n_space_divy][n_space_divx][3])
{
      // cout << "smoothfield" << endl;
    auto *ftemp = static_cast<float(*)[n_space_divy][n_space_divx]>(_aligned_malloc(n_space_divz * n_space_divy * n_space_divx * sizeof(float), alignment));
    //  auto *kspread = static_cast<float(*)[3][3]>(_aligned_malloc(3 * 3 * 3 * sizeof(float), alignment));
    fill(reinterpret_cast<float *>(ftemp), reinterpret_cast<float *>(ftemp) + n_cells, 0.f);
    // calculate center of charge field as offsets (-0.5 to 0.5) from cell center
    //   cout << "calculate center of charge field" << endl;
    for (int k = 0; k < n_space_divz; ++k)
        for (int j = 0; j < n_space_divy; ++j)
            for (int i = 0; i < n_space_divx; ++i)
            {
                fc[k][j][i][0] = (fc[k][j][i][0] / (f[k][j][i] + 1.0e-1f));
                fc[k][j][i][1] = (fc[k][j][i][1] / (f[k][j][i] + 1.0e-1f));
                fc[k][j][i][2] = (fc[k][j][i][2] / (f[k][j][i] + 1.0e-1f));
                if (fabs(fc[k][j][i][0])>0.6)  cout << fc[k][j][i][0] << " "<<f[k][j][i] <<" "<<i<<" " <<j<<" "<<k<<" "<<endl;
                /*
                if (fc[k][j][i][1] > 0.5)
                    cout << fc[k][j][i][0] << " ";
                if (fc[k][j][i][0] < -0.5)
                    cout << fc[k][j][i][1] << " " << f[k][j][i] << " ";
                    */
            }
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
                /*
                if (fx0 < 0)
                    cout << " fx0 " << fx0;
                if (fx1 < 0)
                    cout << " fx1 " << fx1 << " " << fx0 << " " << sw;
                if (fy0 < 0)
                    cout << " fy0 " << fy0;
                if (fy1 < 0)
                    cout << " fy1 " << fy1 << " " << fy0 << " " << sw;
                if (fz0 < 0)
                    cout << " fx0 " << fx0;
                if (fz1 < 0)
                    cout << " fz1 " << fz1;
                //*/
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
    for (int k = 0; k < n_space_divz; ++k)
    {
        for (int j = 0; j < n_space_divy; ++j)
        {
            for (int i = 0; i < n_space_divx; ++i)
            {
                f[k][j][i] = ftemp[k][j][i];
                /*
                if (i == 0)
                    f[k][j][i] = 0;
                if (i == n_space_divx - 1)
                    f[k][j][i] = 0;
                if (j == 0)
                    f[k][j][i] = 0;
                if (j == n_space_divy - 1)
                    f[k][j][i] = 0;
                if (k == 0)
                    f[k][j][i] = 0;
                if (k == n_space_divz - 1)
                    f[k][j][i] = 0;
                    */
            }
        }
    }
    _aligned_free(ftemp);
}

void smoothscalarfieldfft(float f[n_space_divz][n_space_divy][n_space_divx],
                          float fc[n_space_divz][n_space_divy][n_space_divx][3])
{
    // calculate center of charge field
    //   cout << "calculate center of charge field" << endl;

    for (int i = 0; i < n_space_divx; i += 1)
        for (int j = 0; j < n_space_divy; j += 1)
            for (int k = 0; k < n_space_divz; k += 1)
            {
                // int n = i * n_space_divy * n_space_divz + j * n_space_divz + k;
                // Print out the center of charge  grid values
                //                  cout << np[p][k][j][i] << " ";
                fc[k][j][i][0] = (fc[k][j][i][0] / (f[k][j][i] + 1.0e-5f) + (float)i) / (float)n_space_divx - 0.5f;
                fc[k][j][i][1] = (fc[k][j][i][1] / (f[k][j][i] + 1.0e-5f) + (float)j) / (float)n_space_divy - 0.5f;
                fc[k][j][i][2] = (fc[k][j][i][2] / (f[k][j][i] + 1.0e-5f) + (float)k) / (float)n_space_divz - 0.5f;
            }
            //      cout << endl;

#pragma omp barrier
    //  cout << "define NFFT plan" << endl;
    nfftf_plan plan; // Define the NFFT plan

    //  cout << "init NFFT plan" << endl;// Memory allocation is completely done by the init routine.
    nfftf_init_3d(&plan, n_space_divx, n_space_divy, n_space_divz, n_cells);

    //    cout << "fill NFFT plan array with values" << endl;
    for (size_t n = 0; n < n_cells; n++)
    {
        plan.f[n][0] = (reinterpret_cast<float *>(f))[n];
        plan.f[n][1] = 0;
    }

    // cout << "fill NFFT plan x with values" << endl;
    for (size_t n = 0; n < n_cells * 3; n++)
        plan.x[n] = (reinterpret_cast<float *>(fc))[n];

    // cout << "nfft precompute: " << p << endl;
    if (plan.flags & PRE_ONE_PSI)
        nfftf_precompute_one_psi(&plan);

    // cout << "nfft check: ";
    const char *check_error_msg = nfftf_check(&plan);
    if (check_error_msg != 0)
    {
        printf("Invalid nfft parameter: %s\n", check_error_msg);
    }

    //        cout << "Execute NFFT transform plan to get fhat" << endl; //  Execute the forward NFFT transform
    nfftf_adjoint(&plan);
    // make regular equispaced x

    for (int k = 0; k < n_space_divz; k += 1)
        for (int j = 0; j < n_space_divy; j += 1)
            for (int i = 0; i < n_space_divx; i += 1)
            {
                int n = k * n_space_divy * n_space_divz + j * n_space_divz + i;
                plan.x[3 * n] = -0.5 + (float)i / n_space_divx; // p.x[n][0]=x ..,y,z
                plan.x[3 * n + 1] = -0.5 + (float)j / n_space_divy;
                plan.x[3 * n + 2] = -0.5 + (float)k / n_space_divz;
            }

    // cout << "execute the inverse FFTW plan" << endl;  // Execute the inverse FFT
    nfftf_trafo(&plan);
    //        copy back density
    for (unsigned int n = 0; n < n_cells; n++)
        (reinterpret_cast<float *>(f))[n] = plan.f[n][0] / n_cells; // divide by ncell to normalise
    nfftf_finalize(&plan);
}
