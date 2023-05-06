#include "include/traj.h"
void changedt(float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd], float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd], int cdt, par *par)
{
    float inc = 0;
    //   cout << endl<< cdt << " ";
    switch (cdt)
    {
    case 0: // both OK
        break;
    case 1: //
        inc = decf;
        par->dt[0] *= decf;
        par->dt[1] *= decf;
        //     cout << "dt decrease E high B OK \n";
        break;
    case 2:
        //      cout << "dt maintain E too low B OK\n";
        break;
    case 3: // impossible case E too high and too low ..
        inc = 0;
        //      cout << "dt impossible case E too high and too low .. dt\n";
        break;
    case 4:
        inc = decf;
        par->dt[0] *= decf;
        par->dt[1] *= decf;
        //       cout << "dt decrease B exceeded E OK\n";
        break;
    case 5:
        inc = decf;
        par->dt[0] *= decf;
        par->dt[1] *= decf;
        //     cout << "dt decrease B exceeded and E exceeded\n";
        break;
    case 6:
        inc = decf;
        par->dt[0] *= decf;
        par->dt[1] *= decf;
        //cout << "dt decrease B exceeded and E too low\n";
        break;
    case 7: // impossible case E too high and too low ..
        break;
    case 8:
        inc = 0;
        //      cout << "dt maintain B too low E OK\n";
        break;
    case 9:
        inc = decf;
        par->dt[0] *= decf;
        par->dt[1] *= decf;
        //    cout << "dt decrease B too low E too high \n";
        break;
    case 10:
        inc = incf;
        par->dt[0] *= incf;
        par->dt[1] *= incf;
        //    cout << "dt: increase B too low E too low\n";
        break;
    default:
        cout << "error cdt" << endl;
        break;
    }
    if (inc == 0)
        return;
#pragma omp parallel num_threads(2)
    {
        int p = omp_get_thread_num();
#pragma omp parallel sections
        {
#pragma omp section
#pragma omp parallel for simd
            for (int n = 0; n < par->n_part[p]; n++)
                pos0x[p][n] = pos1x[p][n] - (pos1x[p][n] - pos0x[p][n]) * inc;

#pragma omp section
#pragma omp parallel for simd
            for (int n = 0; n < par->n_part[p]; n++)
                pos0y[p][n] = pos1y[p][n] - (pos1y[p][n] - pos0y[p][n]) * inc;

#pragma omp section
#pragma omp parallel for simd
            for (int n = 0; n < par->n_part[p]; n++)
                pos0z[p][n] = pos1z[p][n] - (pos1z[p][n] - pos0z[p][n]) * inc;
        }
    }
}