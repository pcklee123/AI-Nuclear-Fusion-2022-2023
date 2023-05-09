#define RamDisk // whether to use RamDisk if no ramdisk files will be in temp directory
#define maxcells 32
#define cldevice 0
#define sphere          // do hot spot  problem
                        // #define cylinder //do hot rod problem
#define Temp_e 1e7      // in Kelvin
#define Temp_d 1e7      // in Kelvin
constexpr float f1 = 1; // make bigger to make smaller time steps
constexpr float f2 = f1 * 2;
constexpr float incf = 1.2; // increment
constexpr float decf = 0.8; // decrement factor
constexpr int n_space = 64;                                      // must be 2 to power of n
constexpr float R_s = n_space/4;    // LPF smoothing radius
constexpr float r0_f = 8;   //  radius of sphere or cylinder
// The maximum expected E and B fields. If fields go beyond this, the the time step, cell size etc will be wrong. Should adjust and recalculate.
//  maximum expected magnetic field
constexpr float Bmax0 = 10;  // in T
constexpr float Emax0 = 1e6; // 1e11V/m is approximately interatomic E field -extremely large fields implies poor numerical stability
constexpr float nback = 2;   // background particles per cell - improves stability
constexpr float Bz0 = 0.001; // in T
constexpr float Ez0 = 0;
constexpr float a0 = 0.1e-3;        // typical dimensions of a cell in m
constexpr float target_part = 1e11; // 3.5e22 particles per m^3 per torr of ideal gas. 7e22 electrons for 1 torr of deuterium

// technical parameters

constexpr int n_partd = n_space * n_space * n_space * nback * 2; // must be 2 to power of n
constexpr int n_parte = n_partd;
// Te 1e7,Td 1e7,B 0.1,E 1e8,nback 64, a0 0.1e-3,part 1e10,nspace 32 npartd *4 sphere, r1=1.8
// a sphere 0.4 mm radius with 1e24*4/3*pi()*0.4^3 *1.6e-19C E on surface =2.4e12Vm-1 if all electrons have left.
// r0=8*a0 Te 1e7,Td 1e7,B 100,E 1e10,nback 64, a0 1e-3,part 1e15,nspace 64 npartd *4 cylinder
constexpr unsigned int ncoeff = 8;

constexpr int n_output_part = (n_partd > 9369) ? 9369 : n_partd; // maximum number of particles to output to file
// const int nprtd=floor(n_partd/n_output_part);

constexpr int ndatapoints = 300; // total number of time steps to calculate
constexpr int nc = 30;           // number of times to calculate E and B between printouts
constexpr int md_me = 60;       // ratio of electron speed/deuteron speed at the same KE. Used to calculate electron motion more often than deuteron motion

#define Hist_n 1024
#define Hist_max Temp_e / 11600 * 60 // in eV Kelvin to eV is divide by 11600

#define trilinon_
#define Uon_  // whether to calculate the electric (V) potential and potential energy (U). Needs Eon to be enabled.
#define Eon_  // whether to calculate the electric (E) field
#define Bon_  // whether to calculate the magnetic (B) field
#define EFon_ // whether to apply electric force
#define BFon_ // whether to apply magnetic force
#define printDensity
#define printParticles
// #define printV //print out V
#define printB // print out B field
#define printE // print out E field
// #define FileIn //whether to load from input file (unused)

constexpr float r_part_spart = target_part / n_partd; // 1e12 / n_partd; // ratio of particles per tracked "super" particle
// ie. the field of N particles will be multiplied by (1e12/N), as if there were 1e12 particles

constexpr int n_space_divx = n_space;
constexpr int n_space_divy = n_space;
constexpr int n_space_divz = n_space;
constexpr int n_space_divx2 = n_space_divx * 2;
constexpr int n_space_divy2 = n_space_divy * 2;
constexpr int n_space_divz2 = n_space_divz * 2;
constexpr int n_cells = n_space_divx * n_space_divy * n_space_divz;
constexpr int n_cells8 = n_cells * 8;
// physical "constants"
constexpr float kb = 1.38064852e-23;       // m^2kss^-2K-1
constexpr float e_charge = 1.60217662e-19; // C
constexpr float ev_to_j = e_charge;
constexpr float e_mass = 9.10938356e-31;
constexpr float e_charge_mass = e_charge / e_mass;
constexpr float kc = 8.9875517923e9;         // kg m3 s-2 C-2
constexpr float epsilon0 = 8.8541878128e-12; // F m-1
constexpr float pi = 3.1415926536;
constexpr float u0 = 4e-7 * pi;

constexpr int ncalc0[2] = {md_me, 1};
constexpr int qs[2] = {-1, 1}; // Sign of charge
constexpr int mp[2] = {1, 1835 * 2};
struct par // useful parameters
{
    float dt[2]; // time step electron,deuteron
    float Emax = Emax0;
    float Bmax = Bmax0;
    float nt[2];                                                                                                             // total number of particles
    float KEtot[2];                                                                                                         // Total KE of particles
    float posL[3] = {-a0 * (n_space_divx - 1) / 2.0f, -a0 *(n_space_divy - 1.0) / 2.0, -a0 *(n_space_divz - 1.0) / 2.0};    // Lowest position of cells (x,y,z)
    float posH[3] = {a0 * (n_space_divx - 1) / 2.0f, a0 *(n_space_divy - 1.0) / 2.0, a0 *(n_space_divz - 1.0) / 2.0};       // Highes position of cells (x,y,z)
    float posL_1[3] = {-a0 * (n_space_divx - 3) / 2.0f, -a0 *(n_space_divy - 3.0) / 2.0, -a0 *(n_space_divz - 3.0) / 2.0};  // Lowest position of cells (x,y,z)
    float posH_1[3] = {a0 * (n_space_divx - 3) / 2.0f, a0 *(n_space_divy - 3.0) / 2.0, a0 *(n_space_divz - 3.0) / 2.0};     // Highes position of cells (x,y,z)
    float posL_15[3] = {-a0 * (n_space_divx - 4) / 2.0f, -a0 *(n_space_divy - 4.0) / 2.0, -a0 *(n_space_divz - 4.0) / 2.0}; // Lowest position of cells (x,y,z)
    float posH_15[3] = {a0 * (n_space_divx - 4) / 2.0f, a0 *(n_space_divy - 4.0) / 2.0, a0 *(n_space_divz - 4.0) / 2.0};    // Highes position of cells (x,y,z)

    float posL2[3] = {-a0 * n_space_divx, -a0 *n_space_divy, a0 *n_space_divz};
    float dd[3] = {a0, a0, a0}; // cell spacing (x,y,z)

    unsigned int n_space_div[3] = {n_space_divx, n_space_divy, n_space_divz};
    unsigned int n_space_div2[3] = {n_space_divx2, n_space_divy2, n_space_divz2};
    unsigned int n_part[3] = {n_parte, n_partd, n_parte + n_partd};
    float UE = 0;
    float UB = 0;
    // for tnp
    float Ecoef[2] = {0, 0};
    float Bcoef[2] = {0, 0};
    unsigned int ncalcp[2] = {md_me, 1};
    unsigned int n_partp[2] = {n_parte, n_partd}; // 0,number of "super" electrons, electron +deuteriom ions, total
    unsigned int cl_align = 4096;
};

struct field // fields
{
};

struct particles // particles
{
};
