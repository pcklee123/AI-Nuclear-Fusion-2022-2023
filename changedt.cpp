#include "include/traj.h"
void changedt(float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd], float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd], int n_part[3], int cdt,par *par)
{
    int inc=3;
    switch (cdt)
    {
    case 0: // both OK
        break;
    case 1: // E exceeded B OK decrease dt
        inc = 0;
        par->dt[0] /= 2;
        par->dt[1] /= 2;
        // ncalc[0] *= 2;  ncalc[1] *= 2;
        par->Emax *= 2;
        //         par->Bmax *= 1.2;
        cout << "dt decreased E too high \n";
        break;
    case 2: // E too low B OK increase dt
        inc = 1;
        par->dt[0] *= 2;
        par->dt[1] *= 2;
        par->Emax /= 2;
        cout << "dt E too low B OK increase dt\n";
        break;
    case 3: // impossible case E too high and too low ..
        inc = 3;
        cout << "dt impossible case E too high and too low .. dt\n";
        break;
    case 4: // B exceeded E OK decrease dt
        inc = 0;
        par->dt[0] /= 2;
        par->dt[1] /= 2;
        par->Bmax *= 2;
        cout << "dt B exceeded E OK decrease dt\n";
        break;
    case 5: // B exceeded and E exceeded decrease dt
        inc = 0;
        par->dt[0] /= 2;
        par->dt[1] /= 2;
        par->Emax *= 2;
        par->Bmax *= 2;
        cout << "dt B exceeded and E exceeded decrease dt\n";
        break;
    case 6: // B exceeded and E too low decrease dt
        inc = 0;
        par->dt[0] /= 2;
        par->dt[1] /= 2;
        par->Emax /= 2;
        par->Bmax *= 2;
        cout << "dt  B exceeded and E too low decrease dt\n";
        break;
    case 7: // impossible case E too high and too low ..
        break;
    case 8: // B too low E ok increase dt
        inc = 1;
        par->dt[0] *= 2;
        par->dt[1] *= 2;
        par->Bmax /= 2;
        cout << "dt B too low increase dt E ok\n";
        break;
    case 9: // B too low E too high decrease dt
        inc = 0;
        par->dt[0] /= 2;
        par->dt[1] /= 2;
        par->Emax *= 2;
        par->Bmax /= 2;
        cout << "dt  B too low E too high decrease\n";
        break;
    case 10: // B too low E too low increase dt
        inc = 1;
        par->dt[0] *= 2;
        par->dt[1] *= 2;
        // ncalc[0] *= 2;
        //  ncalc[1] *= 2;
        par->Emax /= 2;
        par->Bmax /= 2;
        cout << "dt  B too low E too low increase dt\n";
        break;
    }
    switch (inc)
    {
    case 0: // recalculate pos0 for time step new time step of half
        for (int p = 0; p < 2; p++)
        {
            for (int n = 0; n < n_part[p]; n++)
            {
                pos0x[p][n] = (pos1x[p][n] + pos0x[p][n]) * 0.5;
                pos0y[p][n] = (pos1y[p][n] + pos0y[p][n]) * 0.5;
                pos0z[p][n] = (pos1z[p][n] + pos0z[p][n]) * 0.5;
            }
        }
        break;

    case 1: // increase dt
        for (int p = 0; p < 2; p++)
        {
            for (int n = 0; n < n_part[p]; n++)
            {
                pos0x[p][n] -= (pos1x[p][n] - pos0x[p][n]);
                pos0y[p][n] -= (pos1y[p][n] - pos0y[p][n]);
                pos0z[p][n] -= (pos1z[p][n] - pos0z[p][n]);
            }
        }
        break;
    default:
        break;
    }
}