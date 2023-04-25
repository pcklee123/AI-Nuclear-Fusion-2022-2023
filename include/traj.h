#ifndef TRAJ_H_INCLUDED
#define TRAJ_H_INCLUDED
#include "traj_physics.h"
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <iostream>
#include <omp.h>
#include <string>
#include <filesystem>
#include <CL/opencl.hpp>
// #include <vtk/vtksys/Configure.hxx>
#include <vtk/vtkSmartPointer.h>
#include <vtk/vtkFloatArray.h>
#include <vtk/vtkDoubleArray.h>
#include <vtk/vtkPolyData.h>
#include <vtk/vtkInformation.h>
#include <vtk/vtkTable.h>

#include <vtk/vtkDelimitedTextWriter.h>

#include <vtk/vtkZLibDataCompressor.h>
#include <vtk/vtkXMLImageDataWriter.h>
#include <vtk/vtkXMLPolyDataWriter.h>
#include <vtk/vtkImageData.h>
#include <vtk/vtkPointData.h>
#include <complex>
// #include <infft.h>

#include <nfft3.h>
// #include "nfft3mp.h"
#include <fftw3.h>

using namespace std;
extern cl::Context context_g;
extern cl::Device default_device_g;
extern cl::Program program_g;
extern string outpath;
// #ifdef RamDisk // save file info - initialize filepath
const string outpath1 = "R:\\Temp\\out\\";
#if !defined(_WIN32) && (defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__)))
/* UNIX-style OS. ------------------------------------------- */
const string outpath2 = std::filesystem::temp_directory_path().string() + "/out/";
#else
const string outpath2 = std::filesystem::temp_directory_path().string() + "out/";
#endif
static int nthreads;
constexpr int alignment = 64; // 512 bits / 8 bits per byte = 64 bytes

class Time
{
private:
        vector<chrono::_V2::system_clock::time_point> marks;

public:
        void mark();
        float elapsed();
        float replace();
};
class Log
{
private:
        ofstream log_file;
        bool firstEntry = true; // Whether the next item to print is the first item in the line
public:
        Log();
        template <class T>
        void write(T text, bool flush = false)
        {
                if (!firstEntry)
                        log_file << ",";
                firstEntry = false;
                log_file << text;
                if (flush)
                        log_file.flush();
        }
        void newline();
        void close();
};
static Time timer;
static Log logger;
void log_entry(int i_time, int ntime, int cdt, int total_ncalc[2], float dt[2], double t, int nt[2], float KEtot[2], float U[2]);
void log_headers();
void save_vti_c2(string filename, int i,
                 unsigned int n_space_div[3], float posl[3], float dd[3], uint64_t num, int ncomponents, double t,
                 float data1[3][n_space_divz2][n_space_divy2][n_space_divz2], string typeofdata, int bytesperdata);
void save_vti_c(string filename, int i,
                unsigned int n_space_div[3], float posl[3], float dd[3], uint64_t num, int ncomponents, double t,
                float data1[][n_space_divz][n_space_divy][n_space_divz], string typeofdata, int bytesperdata);
// void save_vti(string filename, int i, unsigned int n_space_div[3], float posl[3], float dd[3], uint64_t num, int ncomponents, double t, float data[n_space_divz][n_space_divy][n_space_divz], string typeofdata, int sizeofdata);
void save_pvd(string filename, int ndatapoints);
// void save_vtp(string filename, int i, uint64_t num, int ncomponents, double t, const char *data, const char *points);
void save_vtp(string filename, int i, uint64_t num, int ncomponents, double t, float data[2][n_output_part], float points[2][n_output_part][3]);
void set_initial_pos_vel(int n_part_types, int n_particles, float *pos0, float *pos1, float *sigma, int *q, int *m, int *nt);
void cl_start();
void cl_set_build_options(float posL[3], float posH[3], float dd[3]);

void tnp(float *Ea1, float *Ba1, float *pos0x, float *pos0y, float *pos0z, float *pos1x, float *pos1y, float *pos1z,
         float cf[2],
         unsigned int ci[2]);
// void get_precalc_r3(float precalc_r3[3][n_space_divz2][n_space_divy2][n_space_divx2], float dd[3]);
int calcEBV(float V[n_space_divz][n_space_divy][n_space_divx],
            float E[3][n_space_divz][n_space_divy][n_space_divx], float B[3][n_space_divz][n_space_divy][n_space_divx],
            float Ee[3][n_space_divz][n_space_divy][n_space_divx], float Be[3][n_space_divz][n_space_divy][n_space_divx],
            float npt[n_space_divz][n_space_divy][n_space_divx], float jc[3][n_space_divz][n_space_divy][n_space_divx],
            float dd[3], float Emax, float Bmax);

void save_files(int i_time, unsigned int n_space_div[3], float posL[3], float dd[3], double t,
                float np[2][n_space_divz][n_space_divy][n_space_divx], float currentj[2][3][n_space_divz][n_space_divy][n_space_divx],
                float V[n_space_divz][n_space_divy][n_space_divx],
                float E[3][n_space_divz][n_space_divy][n_space_divx], float B[3][n_space_divz][n_space_divy][n_space_divx],
                float KE[2][n_output_part], float posp[2][n_output_part][3]);
void sel_part_print(int n_part[3],
                    float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd],
                    float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd],
                    float posp[2][n_output_part][3], float KE[2][n_output_part],
                    int m[2][n_partd], float dt[2]);

void get_densityfields(float currentj[2][3][n_space_divz][n_space_divy][n_space_divx],
                       float np[2][n_space_divz][n_space_divy][n_space_divx],
                       float npt[n_space_divz][n_space_divy][n_space_divx],
                       int nt[2], float KEtot[2], float posL[3], float posH[3], float dd[3],
                       float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd],
                       float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd],
                       int q[2][n_partd], float dt[2], int n_part[3],
                       float jc[3][n_space_divz][n_space_divy][n_space_divz]);
void calc_trilin_constants(float E[3][n_space_divz][n_space_divy][n_space_divx],
                           float Ea[n_space_divz][n_space_divy][n_space_divx][3][ncoeff],
                           float dd[3], float posL[3]);

void changedt(float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd], float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd], int n_part[3]);

void calcU(float V[n_space_divz][n_space_divy][n_space_divx],
           float E[3][n_space_divz][n_space_divy][n_space_divx], float B[3][n_space_divz][n_space_divy][n_space_divx],
           float posx[2][n_partd], float posy[2][n_partd], float posz[2][n_partd],
           float posL[3], float dd[3], int n_part[3], int q[2][n_partd], float out[2]);

void generateParticles(float a0, float r0, int *qs, int *mp, float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd],
                       float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd], int q[2][n_partd], int m[2][n_partd], int *nt);
void generateField(float Ee[3][n_space_divz][n_space_divy][n_space_divx], float Be[3][n_space_divz][n_space_divy][n_space_divx]);
void id_to_cell(int id, int *x, int *y, int *z);
void save_hist(int i_time, double t, int npart, float dt[2], float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd], float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd], int n_part[3]);

void generate_rand_sphere(float a0, float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd],
                          float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd],
                          int q[2][n_partd], int m[2][n_partd], int nt[2], float dt[2]);

void generate_rand_cylinder(float a0, float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd],
                            float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd],
                            int q[2][n_partd], int m[2][n_partd], int nt[2], float dt[2]);
void smoothscalarfield(float f[n_space_divz][n_space_divy][n_space_divx],
                       float fc[n_space_divz][n_space_divy][n_space_divx][3],int s);
#endif // TRAJ_H_INCLUDED
