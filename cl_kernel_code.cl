// Preprocessor things for compilation of tnp
#ifndef XLOW
#define XLOW 0.f
#endif
#ifndef YLOW
#define YLOW 0.f
#endif
#ifndef ZLOW
#define ZLOW 0.f
#endif
#ifndef XHIGH
#define XHIGH 0.f
#endif
#ifndef YHIGH
#define YHIGH 0.f
#endif
#ifndef ZHIGH
#define ZHIGH 0.f
#endif
#ifndef DX
#define DX 0
#endif
#ifndef DY
#define DY 0
#endif
#ifndef DZ
#define DZ 0
#endif
#ifndef NX
#define NX 0
#endif
#ifndef NY
#define NY 0
#endif
#ifndef NZ
#define NZ 0
#endif

void kernel vector_cross_mul(global float *A0, global const float *B0,
                             global const float *C0, global float *A1,
                             global const float *B1, global const float *C1,
                             global float *A2, global const float *B2,
                             global const float *C2) {
  int i = get_global_id(0); // Get index of the current element to be processed
  A0[i] = B1[i] * C2[i] - B2[i] * C1[i]; // Do the operation
  A1[i] = B2[i] * C0[i] - B0[i] * C2[i];
  A2[i] = B0[i] * C1[i] - B1[i] * C0[i];
}

void kernel vector_mul(global float *A, global const float *B,
                       global const float *C) {
  int i = get_global_id(0); // Get index of the current element to be processed
  A[i] = B[i] * C[i];       // Do the operation
}

void kernel vector_muls_addv(global float *A, global const float *B,
                             global const float *C) {
  float Bb = B[0];
  int i = get_global_id(0); // Get index of current element processed
  A[i] = Bb * A[i] + C[i];  // Do the operation
}

void kernel vector_muls(global float *A, global const float *B) {
  float Bb = B[0];
  int i = get_global_id(0); // Get index of current element processed
  A[i] = Bb * A[i];         // Do the operation
}

void kernel vector_mul_complex(global float2 *A, global float2 *B,
                               global float2 *C) {
  int i = get_global_id(0); // Get index of the current element to be processed
  float2 b = B[i], c = C[i];
  A[i] = (float2)(b.s0 * c.s0 - b.s1 * c.s1, b.s0 * c.s1 + b.s1 * c.s0);
}

void kernel tnp_k_implicit(global const float8 *a1,
                           global const float8 *a2, // E, B coeff
                           global float *x0, global float *y0,
                           global float *z0, // prev pos
                           global float *x1, global float *y1,
                           global float *z1, // current pos
                           float Bcoef,
                           float Ecoef, // Bcoeff, Ecoeff
                           const unsigned int n,
                           const unsigned int ncalc, // n, ncalc
                           global float *np, global float *currentj,
                           global int *npi, global int *np_centeri,
                           global int *cji, global int *cj_centeri,
                           global int *q) {

  uint id = get_global_id(0);
  uint prev_idx = UINT_MAX;
  float xprev = x0[id], yprev = y0[id], zprev = z0[id], x = x1[id], y = y1[id],
        z = z1[id];
  float8 temp, pos;
  float r1 = 1.0f;
  float r2 = r1 * r1;
  float8 store0, store1, store2, store3, store4, store5;
  const float Bcoeff = Bcoef / r1;
  const float Ecoeff = Ecoef / r1;
  const float XL = XLOW + 1.5f * DX, YL = YLOW + 1.5f * DY,
              ZL = ZLOW + 1.5f * DZ;
  const float XH = XHIGH - 1.5f * DX, YH = YHIGH - 1.5f * DY,
              ZH = ZHIGH - 1.5f * DZ;

  for (int t = 0; t < ncalc; t++) {

    // if (x <= XLOW || x >= XHIGH || y <= YLOW || y >= YHIGH || z <= ZLOW ||
    //  z >= ZHIGH)
    // break;
    float xy = x * y, xz = x * z, yz = y * z, xyz = x * yz;
    uint idx =
        ((uint)((z - ZLOW) / DZ) * NZ + (uint)((y - YLOW) / DY)) * NY +
        (uint)((x - XLOW) / DX); // round down the cells - this is intentional
    idx *= 3;
    pos = (float8)(1.f, x, y, z, xy, xz, yz, xyz);
    // Is there no better way to do this? Why does float8 not have dot()?
    if (prev_idx != idx) {
      store0 = a1[idx]; // Ex
      store1 = a1[idx + 1];
      store2 = a1[idx + 2];
      store3 = a2[idx]; // Bx
      store4 = a2[idx + 1];
      store5 = a2[idx + 2];
      prev_idx = idx;
    }
    temp = store0 * pos;
    float xE = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store1 * pos;
    float yE = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store2 * pos;
    float zE = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store3 * pos;
    float xP = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store4 * pos;
    float yP = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;
    temp = store5 * pos;
    float zP = temp.s0 + temp.s1 + temp.s2 + temp.s3 + temp.s4 + temp.s5 +
               temp.s6 + temp.s7;

    xP *= Bcoeff;
    yP *= Bcoeff;
    zP *= Bcoeff;
    xE *= Ecoeff;
    yE *= Ecoeff;
    zE *= Ecoeff;

    float xyP = xP * yP, yzP = yP * zP, xzP = xP * zP;
    float xxP = xP * xP, yyP = yP * yP, zzP = zP * zP;
    // float b_det = 1.f / (1.f + xxP + yyP + zzP);
    float b_det = r2 / (r2 + xxP + yyP + zzP);

    float vx = (x - xprev); // / dt -> cancels out in the end
    float vy = (y - yprev);
    float vz = (z - zprev);

    xprev = x;
    yprev = y;
    zprev = z;

    float vxxe = vx + xE, vyye = vy + yE, vzze = vz + zE;
    // x += vx + b_det * (-vx * (yyP + zzP) + vyye * (zP + xyP) + vzze * (xzP -
    // yP) + (1.f + xxP) * xE); y += vy + b_det * (vxxe * (xyP - zP) -  vy *
    // (xxP + zzP) + vzze * (xP + yzP) + (1.f + yyP) * yE); z += vz + b_det *
    // (vxxe * (yP + xzP) + vyye * (yzP - xP) -  vz * (xxP + yyP) + (1.f * zzP)
    // * zE);
    //   do fma
    x += fma(b_det,
             fma(-vx, yyP + zzP,
                 fma(vyye, zP + xyP, fma(vzze, xzP - yP, fma(xxP, xE, xE)))),
             vx);
    y += fma(b_det,
             fma(vxxe, xyP - zP,
                 fma(-vy, xxP + zzP, fma(vzze, xP + yzP, fma(yyP, yE, yE)))),
             vy);
    z += fma(b_det,
             fma(vxxe, yP + xzP,
                 fma(vyye, yzP - xP, fma(-vz, xxP + yyP, fma(zzP, zE, zE)))),
             vz);
    xprev = x > XL ? xprev : XL;
    xprev = x < XH ? xprev : XH;
    yprev = y > YL ? yprev : YL;
    yprev = y < YH ? yprev : YH;
    zprev = z > ZL ? zprev : ZL;
    zprev = z < ZH ? zprev : ZH;
    q[id] = (x > XL & x<XH & y> YL & y<YH & z> ZL & z < ZH) ? q[id] : 0;
    x = x > XL ? x : XL;
    x = x < XH ? x : XH;
    y = y > YL ? y : YL;
    y = y < YH ? y : YH;
    z = z > ZL ? z : ZL;
    z = z < ZH ? z : ZH;

  }

  uint k = round((z - ZLOW) / DZ);
  uint j = round((y - YLOW) / DY);
  uint i = round((x - XLOW) / DX);
  int offsetx = (x / DX - XLOW / DX - i) * 256.0f;
  int offsety = (y / DY - YLOW / DY - j) * 256.0f;
  int offsetz = (z / DZ - ZLOW / DZ - k) * 256.0f;
  uint idx00 = k * NY * NX + j * NX + i;
  uint idx01 = idx00 + NZ * NY * NX;
  uint idx02 = idx01 + NZ * NY * NX;
  atomic_add(&npi[idx00], q[id]);
  atomic_add(&cji[idx00], ((x - xprev) * 65536.0f) * q[id]);
  atomic_add(&cji[idx01], ((y - yprev) * 65536.0f) * q[id]);
  atomic_add(&cji[idx02], ((z - zprev) * 65536.0f) * q[id]);

  np[idx00] = npi[idx00];
  currentj[idx00] = cji[idx00] / 65536.0f;
  currentj[idx01] = cji[idx01] / 65536.0f;
  currentj[idx02] = cji[idx02] / 65536.0f;
  x0[id] = xprev;
  y0[id] = yprev;
  z0[id] = zprev;
  x1[id] = x;
  y1[id] = y;
  z1[id] = z;
}

void kernel density(
                    global float *x0, global float *y0,
                    global float *z0, // prev pos
                    global float *x1, global float *y1,
                    global float *z1, // current pos
                    global float *np, global float *currentj, global int *npi,
                    global int *np_centeri, global int *cji,
                    global int *cj_centeri, global int *q) {
  const float XL = XLOW + 1.5f * DX, YL = YLOW + 1.5f * DY,
              ZL = ZLOW + 1.5f * DZ;
  const float XH = XHIGH - 1.5f * DX, YH = YHIGH - 1.5f * DY,
              ZH = ZHIGH - 1.5f * DZ;

  uint id = get_global_id(0);
  float xprev = x0[id], yprev = y0[id], zprev = z0[id], x = x1[id], y = y1[id],
        z = z1[id];
  xprev = x > XL ? xprev : XL;
  xprev = x < XH ? xprev : XH;
  yprev = y > YL ? yprev : YL;
  yprev = y < YH ? yprev : YH;
  zprev = z > ZL ? zprev : ZL;
  zprev = z < ZH ? zprev : ZH;
  q[id] = (x > XL & x<XH & y> YL & y<YH & z> ZL & z < ZH) ? q[id] : 0;
  x = x > XL ? x : XL;
  x = x < XH ? x : XH;
  y = y > YL ? y : YL;
  y = y < YH ? y : YH;
  z = z > ZL ? z : ZL;
  z = z < ZH ? z : ZH;

  uint k = round((z - ZLOW) / DZ);
  uint j = round((y - YLOW) / DY);
  uint i = round((x - XLOW) / DX);
  //  int offsetx = (x / DX - XLOW / DX - i) * 256.0f;
  //  int offsety = (y / DY - YLOW / DY - j) * 256.0f;
  //  int offsetz = (z / DZ - ZLOW / DZ - k) * 256.0f;
  uint idx00 = k * NY * NX + j * NX + i;
  uint idx01 = idx00 + NZ * NY * NX;
  uint idx02 = idx01 + NZ * NY * NX;
  atomic_add(&npi[idx00], q[id]);
  atomic_add(&cji[idx00], ((x - xprev) * 65536.0f) * q[id]);
  atomic_add(&cji[idx01], ((y - yprev) * 65536.0f) * q[id]);
  atomic_add(&cji[idx02], ((z - zprev) * 65536.0f) * q[id]);

  np[idx00] = npi[idx00];
  currentj[idx00] = cji[idx00] / 65536.0f;
  currentj[idx01] = cji[idx01] / 65536.0f;
  currentj[idx02] = cji[idx02] / 65536.0f;
}