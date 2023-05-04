#include "include/traj.h"
void changedt(float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd], float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd], int cdt, par *par)
{
    int inc = 3;
 //   cout << endl<< cdt << " ";
    switch (cdt)
    {
    case 0: // both OK
        break;
    case 1: //
        inc = 0;
        par->dt[0] /= 2;
        par->dt[1] /= 2;
   //     cout << "dt decrease E high B OK \n";
        break;
    case 2:
  //      cout << "dt maintain E too low B OK\n";
        break;
    case 3: // impossible case E too high and too low ..
        inc = 3;
  //      cout << "dt impossible case E too high and too low .. dt\n";
        break;
    case 4:
        inc = 0;
        par->dt[0] /= 2;
        par->dt[1] /= 2;
 //       cout << "dt decrease B exceeded E OK\n";
        break;
    case 5:
        inc = 0;
        par->dt[0] /= 2;
        par->dt[1] /= 2;
   //     cout << "dt decrease B exceeded and E exceeded\n";
        break;
    case 6:
        inc = 0;
        par->dt[0] /= 2;
        par->dt[1] /= 2;
        cout << "dt decrease B exceeded and E too low\n";
        break;
    case 7: // impossible case E too high and too low ..
        break;
    case 8:
  //      cout << "dt maintain B too low E OK\n";
        break;
    case 9:
        inc = 0;
        par->dt[0] /= 2;
        par->dt[1] /= 2;
    //    cout << "dt decrease B too low E too high \n";
        break;
    case 10:
        inc = 1;
        par->dt[0] *= 2;
        par->dt[1] *= 2;
    //    cout << "dt: increase B too low E too low\n";
        break;
    default:
        break;
    }
    switch (inc)
    {
    case 0: // recalculate pos0 for time step new time step of half
        for (int p = 0; p < 2; p++)
        {
            for (int n = 0; n < par->n_part[p]; n++)
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
            for (int n = 0; n < par->n_part[p]; n++)
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