#include "include/traj.h"

void save_files(int i_time, unsigned int n_space_div[3], float posL[3], float dd[3], double t,
                float np[2][n_space_divz][n_space_divy][n_space_divx], float currentj[2][3][n_space_divz][n_space_divy][n_space_divx],
                float V[n_space_divz][n_space_divy][n_space_divx],
                float E[3][n_space_divz][n_space_divy][n_space_divx], float B[3][n_space_divz][n_space_divy][n_space_divx],
                float KE[2][n_output_part], float posp[2][n_output_part][3])
{
#ifdef printDensity
  save_vti_c("Ne", i_time, n_space_div, posL, dd, n_cells, 1, t, &np[0], "Float32", sizeof(float));
  save_vti_c("je", i_time, n_space_div, posL, dd, n_cells, 3, t, currentj[0], "Float32", sizeof(float));
#endif
#ifdef printV
  save_vti_c("V", i_time, n_space_div, posL, dd, n_cells, 1, t, V, "Float32", sizeof(float));
#endif
#ifdef printE
  save_vti_c("E", i_time, n_space_div, posL, dd, n_cells, 3, t, E, "Float32", sizeof(float));
#endif
#ifdef printB
  save_vti_c("B", i_time, n_space_div, posL, dd, n_cells, 3, t, B, "Float32", sizeof(float));
#endif
#ifdef printParticles
  save_vtp("e", i_time, n_output_part, 0, t, KE, posp);
  save_vtp("d", i_time, n_output_part, 1, t, KE, posp);
#endif
}
void save_hist(int i_time, double t, int npart, float dt[2], float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd], float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd], int n_part[3])
{
  // Create the vtkTable object
  vtkSmartPointer<vtkTable> table = vtkSmartPointer<vtkTable>::New();
  // Create the histogram arrays
  vtkSmartPointer<vtkDoubleArray> energyArray = vtkSmartPointer<vtkDoubleArray>::New();
  energyArray->SetName("Energy(eV)");
  vtkSmartPointer<vtkDoubleArray> electronHistArray = vtkSmartPointer<vtkDoubleArray>::New();
  electronHistArray->SetName("Electron KE Histogram");
  vtkSmartPointer<vtkDoubleArray> ionHistArray = vtkSmartPointer<vtkDoubleArray>::New();
  ionHistArray->SetName("Ion KE Histogram");

  long KEhist[2][Hist_n];
  memset(KEhist, 0, sizeof(KEhist));
  float coef[2];
  for (int p = 0; p < 2; ++p)
  {
    coef[p] = 0.5 * (float)mp[p] * (float)Hist_n / (e_charge_mass * dt[p] * dt[p] * (float)Hist_max);
    for (int i = 0; i < n_part[p]; ++i)
    {
      float dx = pos1x[p][i] - pos0x[p][i];
      float dy = pos1y[p][i] - pos0y[p][i];
      float dz = pos1z[p][i] - pos0z[p][i];
      unsigned int index = (int)floor(coef[p] * (dx * dx + dy * dy + dz * dz));
      if (index < Hist_n)
        //        index = Hist_n - 1;
        //   if (index < 0)
        //     cout << "error index<0"<<(0.5 * (float)mp[p] * (dx * dx + dy * dy + dz * dz) * (float)Hist_n/ (e_charge_mass * dt[p] * dt[p]*(float)Hist_max))<< endl;
        KEhist[p][index]++;
    }
  }
  // Add the histogram values to the arrays
  for (int i = 0; i < Hist_n; ++i)
  {
    energyArray->InsertNextValue(((double)(i + 0.5) * (double)Hist_max) / (double)(Hist_n));
    electronHistArray->InsertNextValue((double)(KEhist[0][i] + 1));
    ionHistArray->InsertNextValue((double)(KEhist[1][i] + 1));
  }

  // Add the histogram arrays to the table
  table->AddColumn(energyArray);
  table->AddColumn(electronHistArray);
  table->AddColumn(ionHistArray);

  // Write the table to a file
  vtkSmartPointer<vtkDelimitedTextWriter> writer = vtkSmartPointer<vtkDelimitedTextWriter>::New();
  writer->SetFileName((outpath + "KEhist_" + to_string(i_time) + ".csv").c_str());
  writer->SetInputData(table);
  writer->Write();
}

/**
 * This corrects the order of dimensions for view in paraview, as opposed to save_vti which prints the raw data.
 */
void save_vti_c(string filename, int i,
                unsigned int n_space_div[3], float posl[3], float dd[3], uint64_t num, int ncomponents, double t,
                float data1[][n_space_divz][n_space_divy][n_space_divz], string typeofdata, int bytesperdata)
{
  if (ncomponents > 3)
  {
    cout << "Error: Cannot write file " << filename << " - too many components" << endl;
    return;
  }
  int xi = (n_space_divx - 1) / maxcells + 1;
  int yj = (n_space_divy - 1) / maxcells + 1;
  int zk = (n_space_divz - 1) / maxcells + 1;
  int nx = n_space_div[0] / xi;
  int ny = n_space_div[1] / yj;
  int nz = n_space_div[2] / zk;
  // vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
  vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New(); // Create the vtkImageData object
  imageData->SetDimensions(nx, ny, nz);                                           // Set the dimensions of the image data
  imageData->SetSpacing(dd[0] * xi, dd[1] * yj, dd[2] * zk);
  imageData->SetOrigin(posl[0], posl[1], posl[2]); // Set the origin of the image data
  imageData->AllocateScalars(VTK_FLOAT, ncomponents);
  imageData->GetPointData()->GetScalars()->SetName(filename.c_str());
  float *data2 = static_cast<float *>(imageData->GetScalarPointer()); // Get a pointer to the density field array

  for (int k = 0; k < nz; ++k)
  {
    for (int j = 0; j < ny; ++j)
    {
      for (int i = 0; i < nx; ++i)
      {
        for (int c = 0; c < ncomponents; ++c)
          data2[(k * ny + j) * nx * ncomponents + i * ncomponents + c] = data1[c][k * zk][j * yj][i * xi];
      }
    }
  }

  vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New(); // Create the vtkXMLImageDataWriter object
  writer->SetFileName((outpath + filename + "_" + to_string(i) + ".vti").c_str());               // Set the output file name                                                                     // Set the time value
  writer->SetDataModeToBinary();
  // writer->SetCompressorTypeToLZ4();
  writer->SetCompressorTypeToZLib(); // Enable compression
  writer->SetCompressionLevel(9);    // Set the level of compression (0-9)
  writer->SetInputData(imageData);   // Set the input image data
  writer->Write();                   // Write the output file
}

void save_vtp(string filename, int i, uint64_t num, int n, double t, float data[2][n_output_part], float points1[2][n_output_part][3])
{
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  vtkSmartPointer<vtkFloatArray> kineticEnergy = vtkSmartPointer<vtkFloatArray>::New();
  kineticEnergy->SetName("KE");
  for (int i = 0; i < num; ++i)
  {
    points->InsertNextPoint(points1[n][i][0], points1[n][i][1], points1[n][i][2]);
    kineticEnergy->InsertNextValue(data[n][i]);
  }
  vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New(); // Create the VTK poly data object
  polyData->SetPoints(points);
  polyData->GetPointData()->AddArray(kineticEnergy);
  // Write the output file
  vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
  writer->SetFileName((outpath + filename + "_" + to_string(i) + ".vtp").c_str());
  writer->SetDataModeToBinary();
  writer->SetCompressorTypeToZLib(); // Enable compression
  writer->SetCompressionLevel(9);    // Set the level of compression (0-9)
  writer->SetInputData(polyData);
  writer->Write();
}
