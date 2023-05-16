#include "include/traj.h"
void tnp(fields *fi, particles *pt, par *par)
{
   unsigned int n0 = n_partd;                 // number of particles ci[0];
   unsigned int n = n_partd * 2;              // both electron and ion
   unsigned int n4 = n_partd * sizeof(float); // number of particles * sizeof(float)
   unsigned int n8 = n * sizeof(float);       // number of particles * sizeof(float)
   unsigned int nc = n_cells * ncoeff * 3;    // trilin constatnts have 8 coefficients 3 components
   unsigned int n_cellsi = n_cells * sizeof(int);
   unsigned int n_cellsf = n_cells * sizeof(float);
   static bool fastIO;
   static bool first = true;
   if (first)
   { // get whether or not we are on an iGPU/similar, and can use certain memmory optimizations
      bool temp;
      default_device_g.getInfo(CL_DEVICE_HOST_UNIFIED_MEMORY, &temp);
      if (temp == true)
         info_file << "Using unified memory: " << temp << " ";
      else
         info_file << "No unified memory: " << temp << " ";
      fastIO = temp;
      //    fastIO = false;
   }
#ifdef BFon_
   par->Bcoef[0] = (float)qs[0] * e_charge_mass / (float)mp[0] * par->dt[0] * 0.5f;
   par->Bcoef[1] = (float)qs[1] * e_charge_mass / (float)mp[1] * par->dt[1] * 0.5f;
#else
   par->Bcoef = {0, 0};
#endif
#ifdef EFon_
   par->Ecoef[0] = par->Bcoef[0] * par->dt[0]; // multiply by dt because of the later portion of cl code
   par->Ecoef[1] = par->Bcoef[1] * par->dt[1]; // multiply by dt because of the later portion of cl code
#else
   par->Ecoef = {0, 0};
#endif
   // cout << " Bconst=" << par->Bcoef[0] << ", Econst=" << par->Ecoef[0] << endl;
   //  create buffers on the device
   /** IMPORTANT: do not use CL_MEM_USE_HOST_PTR if on dGPU **/
   /** HOST_PTR is only used so that memory is not copied, but instead shared between CPU and iGPU in RAM**/
   // Note that special alignment has been given to Ea, Ba, y0, z0, x0, x1, y1 in order to actually do this properly

   // Assume buffers A, B, I, J (Ea, Ba, ci, cf) will always be the same. Then we save a bit of time.
   static cl::Buffer buff_Ea(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_ONLY, sizeof(float) * nc, fastIO ? fi->Ea : NULL);
   static cl::Buffer buff_Ba(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_ONLY, sizeof(float) * nc, fastIO ? fi->Ba : NULL);
   static cl::Buffer buff_np_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_ONLY, sizeof(float) * n_cellsf, fastIO ? fi->np[0] : NULL);
   static cl::Buffer buff_np_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_ONLY, sizeof(float) * n_cellsf, fastIO ? fi->np[1] : NULL);
   static cl::Buffer buff_currentj_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_ONLY, sizeof(float) * n_cellsf * 3, fastIO ? fi->currentj[0] : NULL);
   static cl::Buffer buff_currentj_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_ONLY, sizeof(float) * n_cellsf * 3, fastIO ? fi->currentj[1] : NULL);

   static cl::Buffer buff_npi(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_ONLY, sizeof(int) * n_cellsi, fastIO ? fi->npi : NULL);
   static cl::Buffer buff_cji(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_ONLY, sizeof(int) * n_cellsi * 3, fastIO ? fi->cji : NULL);

   static cl::Buffer buff_np_centeri(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_ONLY, sizeof(int) * n_cellsi * 3, fastIO ? fi->np_centeri : NULL);
   static cl::Buffer buff_cj_centeri(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_ONLY, sizeof(int) * n_cellsi * 3 * 3, fastIO ? fi->cj_centeri : NULL);

   static cl::Buffer buff_x0_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0x[0] : NULL); // x0
   static cl::Buffer buff_y0_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0y[0] : NULL); // y0
   static cl::Buffer buff_z0_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0z[0] : NULL); // z0
   static cl::Buffer buff_x1_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1x[0] : NULL); // x1
   static cl::Buffer buff_y1_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1y[0] : NULL); // y1
   static cl::Buffer buff_z1_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1z[0] : NULL); // z1

   static cl::Buffer buff_q_e(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->q[0] : NULL); // q

   static cl::Buffer buff_x0_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0x[1] : NULL); // x0
   static cl::Buffer buff_y0_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0y[1] : NULL); // y0
   static cl::Buffer buff_z0_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos0z[1] : NULL); // z0
   static cl::Buffer buff_x1_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1x[1] : NULL); // x1
   static cl::Buffer buff_y1_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1y[1] : NULL); // y1
   static cl::Buffer buff_z1_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->pos1z[1] : NULL); // z1

   static cl::Buffer buff_q_i(context_g, (fastIO ? CL_MEM_USE_HOST_PTR : 0) | CL_MEM_READ_WRITE, n4, fastIO ? pt->q[1] : NULL); // q
                                                                                                                                // */
   // cout << "command q" << endl; //  create queue to which we will push commands for the device.

   static cl::CommandQueue queue(context_g, default_device_g);
   cl::Kernel kernel_add = cl::Kernel(program_g, "tnp_k_implicit"); // select the kernel program to run
   // write input arrays to the device
   if (fastIO)
   { // is mapping required? // Yes we might need to map because OpenCL does not guarantee that the data will be shared, alternatively use SVM
     // auto * mapped_buff_x0_e = (float *)queue.enqueueMapBuffer(buff_x0_e, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * n); queue.enqueueUnmapMemObject(buff_x0_e, mapped_buff_x0_e);
   }
   else
   {
      //  cout << "write buffer" << endl;
      queue.enqueueWriteBuffer(buff_Ea, CL_TRUE, 0, sizeof(float) * nc, fi->Ea);
      queue.enqueueWriteBuffer(buff_Ba, CL_TRUE, 0, sizeof(float) * nc, fi->Ba);

      queue.enqueueFillBuffer(buff_npi, 0, 0, n_cellsi);
      queue.enqueueFillBuffer(buff_np_centeri, 0, 0, n_cellsi * 3);
      queue.enqueueFillBuffer(buff_cji, 0, 0, n_cellsi * 3);
      queue.enqueueFillBuffer(buff_cj_centeri, 0, 0, n_cellsi * 3 * 3);

      if (first)
      {
         queue.enqueueWriteBuffer(buff_x0_e, CL_TRUE, 0, n4, pt->pos0x[0]);
         queue.enqueueWriteBuffer(buff_y0_e, CL_TRUE, 0, n4, pt->pos0y[0]);
         queue.enqueueWriteBuffer(buff_z0_e, CL_TRUE, 0, n4, pt->pos0z[0]);
         queue.enqueueWriteBuffer(buff_x1_e, CL_TRUE, 0, n4, pt->pos1x[0]);
         queue.enqueueWriteBuffer(buff_y1_e, CL_TRUE, 0, n4, pt->pos1y[0]);
         queue.enqueueWriteBuffer(buff_z1_e, CL_TRUE, 0, n4, pt->pos1z[0]);

         queue.enqueueWriteBuffer(buff_q_e, CL_TRUE, 0, n4, pt->q[0]);
         // queue.enqueueWriteBuffer(buff_x0_e, CL_TRUE, 0, n4 * 6 * 2, pt->pos0x[0]);
      }
   }
   // return;
   //  set arguments to be fed into the kernel program
   //  cout << "kernel arguments for electron" << endl;

   kernel_add.setArg(0, buff_Ea);                        // the 1st argument to the kernel program Ea
   kernel_add.setArg(1, buff_Ba);                        // Ba
   kernel_add.setArg(2, buff_x0_e);                      // x0
   kernel_add.setArg(3, buff_y0_e);                      // y0
   kernel_add.setArg(4, buff_z0_e);                      // z0
   kernel_add.setArg(5, buff_x1_e);                      // x1
   kernel_add.setArg(6, buff_y1_e);                      // y1
   kernel_add.setArg(7, buff_z1_e);                      // z1
   kernel_add.setArg(8, sizeof(float), &par->Bcoef[0]);  // Bconst
   kernel_add.setArg(9, sizeof(float), &par->Ecoef[0]);  // Econst
   kernel_add.setArg(10, sizeof(int), &par->n_partp[0]); // npart
   kernel_add.setArg(11, sizeof(int), &par->ncalcp[0]);  // ncalc
   kernel_add.setArg(12, buff_np_e);                     // np
   kernel_add.setArg(13, buff_currentj_e);               // current
   kernel_add.setArg(14, buff_npi);                      // npt
   kernel_add.setArg(15, buff_np_centeri);               // npt
   kernel_add.setArg(16, buff_cji);                      // current
   kernel_add.setArg(17, buff_cj_centeri);               // npt
   kernel_add.setArg(18, buff_q_e);                      // q
                                                         // kernel_add.setArg(14, sizeof(int), n_cells);          // ncells
   // cout << "run kernel for electron" << endl;

   // run the kernel
   queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(n0), cl::NullRange);

   queue.finish(); // wait for the end of the kernel program
   queue.enqueueReadBuffer(buff_np_e, CL_TRUE, 0, n_cellsf, fi->np[0]);
   queue.enqueueReadBuffer(buff_currentj_e, CL_TRUE, 0, n_cellsf, fi->currentj[0]);
   // ions next
   if (first)
   {
      queue.enqueueWriteBuffer(buff_x0_i, CL_TRUE, 0, n4, pt->pos0x[1]);
      queue.enqueueWriteBuffer(buff_y0_i, CL_TRUE, 0, n4, pt->pos0y[1]);
      queue.enqueueWriteBuffer(buff_z0_i, CL_TRUE, 0, n4, pt->pos0z[1]);
      queue.enqueueWriteBuffer(buff_x1_i, CL_TRUE, 0, n4, pt->pos1x[1]);
      queue.enqueueWriteBuffer(buff_y1_i, CL_TRUE, 0, n4, pt->pos1y[1]);
      queue.enqueueWriteBuffer(buff_z1_i, CL_TRUE, 0, n4, pt->pos1z[1]);

      queue.enqueueWriteBuffer(buff_q_i, CL_TRUE, 0, n4, pt->q[0]);
   }
   queue.enqueueFillBuffer(buff_npi, 0, 0, n_cellsi);
   queue.enqueueFillBuffer(buff_np_centeri, 0, 0, n_cellsi * 3);
   queue.enqueueFillBuffer(buff_cji, 0, 0, n_cellsi * 3);
   queue.enqueueFillBuffer(buff_cj_centeri, 0, 0, n_cellsi * 3 * 3);
   //  set arguments to be fed into the kernel program
   kernel_add.setArg(0, buff_Ea);                        // the 1st argument to the kernel program Ea
   kernel_add.setArg(1, buff_Ba);                        // Ba
   kernel_add.setArg(2, buff_x0_i);                      // x0
   kernel_add.setArg(3, buff_y0_i);                      // y0
   kernel_add.setArg(4, buff_z0_i);                      // z0
   kernel_add.setArg(5, buff_x1_i);                      // x1
   kernel_add.setArg(6, buff_y1_i);                      // y1
   kernel_add.setArg(7, buff_z1_i);                      // z1
   kernel_add.setArg(8, sizeof(float), &par->Bcoef[1]);  // Bconst
   kernel_add.setArg(9, sizeof(float), &par->Ecoef[1]);  // Econst
   kernel_add.setArg(10, sizeof(int), &par->n_partp[1]); // npart
   kernel_add.setArg(11, sizeof(int), &par->ncalcp[1]);  //
   kernel_add.setArg(12, buff_np_i);                     // npt
   kernel_add.setArg(13, buff_currentj_i);               // current
   kernel_add.setArg(14, buff_npi);                      // npt
   kernel_add.setArg(15, buff_np_centeri);               // npt
   kernel_add.setArg(16, buff_cji);                      // current
   kernel_add.setArg(17, buff_cj_centeri);               // npt
   kernel_add.setArg(18, buff_q_i);                      // q
                                                         // kernel_add.setArg(14, sizeof(int), &n_cells);          // ncells
   // cout << "run kernel for ions" << endl;
   //  run the kernel
   queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(n0), cl::NullRange);
   queue.finish(); // wait for the end of the kernel program
   // read result arrays from the device to main memory
   if (fastIO)
   { // is mapping required?
     // mapped_buff_x0_e = (float *)queue.enqueueMapBuffer(buff_x0_e, CL_TRUE, CL_MAP_READ, 0, sizeof(float) * n); queue.enqueueUnmapMemObject(buff_x0_e, mapped_buff_x0_e);
   }
   else
   {
      queue.enqueueReadBuffer(buff_x0_e, CL_TRUE, 0, n4, pt->pos0x[0]);
      queue.enqueueReadBuffer(buff_y0_e, CL_TRUE, 0, n4, pt->pos0y[0]);
      queue.enqueueReadBuffer(buff_z0_e, CL_TRUE, 0, n4, pt->pos0z[0]);
      queue.enqueueReadBuffer(buff_x1_e, CL_TRUE, 0, n4, pt->pos1x[0]);
      queue.enqueueReadBuffer(buff_y1_e, CL_TRUE, 0, n4, pt->pos1y[0]);
      queue.enqueueReadBuffer(buff_z1_e, CL_TRUE, 0, n4, pt->pos1z[0]);

      queue.enqueueReadBuffer(buff_x0_i, CL_TRUE, 0, n4, pt->pos0x[1]);
      queue.enqueueReadBuffer(buff_y0_i, CL_TRUE, 0, n4, pt->pos0y[1]);
      queue.enqueueReadBuffer(buff_z0_i, CL_TRUE, 0, n4, pt->pos0z[1]);
      queue.enqueueReadBuffer(buff_x1_i, CL_TRUE, 0, n4, pt->pos1x[1]);
      queue.enqueueReadBuffer(buff_y1_i, CL_TRUE, 0, n4, pt->pos1y[1]);
      queue.enqueueReadBuffer(buff_z1_i, CL_TRUE, 0, n4, pt->pos1z[1]);

      queue.enqueueReadBuffer(buff_np_i, CL_TRUE, 0, n_cellsf, fi->np[1]);
      queue.enqueueReadBuffer(buff_currentj_i, CL_TRUE, 0, n_cellsf, fi->currentj[1]);
#pragma omp parallel for simd num_threads(nthreads)
      for (unsigned int i = 0; i < n_cells; i++)
         (reinterpret_cast<float *>(fi->npt))[i] = (reinterpret_cast<float *>(fi->np[0]))[i] + (reinterpret_cast<float *>(fi->np[1]))[i];

#pragma omp parallel for simd num_threads(nthreads)
      for (unsigned int i = 0; i < n_cells * 3; i++)
         (reinterpret_cast<float *>(fi->jc))[i] = (reinterpret_cast<float *>(fi->currentj[0]))[i] + (reinterpret_cast<float *>(fi->currentj[1]))[i];
#pragma omp barrier
   }
   first = false;
}