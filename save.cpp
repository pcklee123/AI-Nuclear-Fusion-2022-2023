#include "include/traj.h"

void save_files(int i_time, double t,
                float np[2][n_space_divz][n_space_divy][n_space_divx], float currentj[2][3][n_space_divz][n_space_divy][n_space_divx],
                float V[n_space_divz][n_space_divy][n_space_divx],
                float E[3][n_space_divz][n_space_divy][n_space_divx], float B[3][n_space_divz][n_space_divy][n_space_divx],
                float KE[2][n_output_part], float posp[2][n_output_part][3], par *par)
{
#pragma omp parallel sections
  {
#ifdef printDensity
#pragma omp section
    save_vti_c("Ne", i_time, 1, t, &np[0], par);
#pragma omp section
    save_vti_c("je", i_time, 3, t, currentj[0], par);
#endif
#ifdef printV
#pragma omp section
    save_vti_c("V", i_time, 1, t, V, par);
#endif
#ifdef printE
#pragma omp section
    save_vti_c("E", i_time, 3, t, E, par);
#endif
#ifdef printB
#pragma omp section
    save_vti_c("B", i_time, 3, t, B, par);
#endif
#ifdef printParticles
#pragma omp section
    save_vtp("e", i_time, n_output_part, t, KE[0], posp[0]);
#pragma omp section
    save_vtp("d", i_time, n_output_part, t, KE[1], posp[1]);
#endif
  }
}

void save_hist(int i_time, double t, int q[2][n_partd], float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd], float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd], par *par)
{
  long KEhist[2][Hist_n];
  memset(KEhist, 0, sizeof(KEhist));
  float coef[2];
  for (int p = 0; p < 2; ++p)
  {
    float KE = 0;
    long nt = 0;
    coef[p] = 0.5 * (float)mp[p] * (float)Hist_n / (e_charge_mass * par->dt[p] * par->dt[p] * (float)Hist_max);
    for (int i = 0; i < par->n_part[p]; ++i)
    {
      float dx = pos1x[p][i] - pos0x[p][i];
      float dy = pos1y[p][i] - pos0y[p][i];
      float dz = pos1z[p][i] - pos0z[p][i];
      float v2 = (dx * dx + dy * dy + dz * dz);
      unsigned int index = (int)floor(coef[p] * v2);
      KE += v2;
      nt += q[p][i];

      if (index >= Hist_n)
        index = Hist_n - 1;
      //   if (index < 0) cout << "error index<0"<<(0.5 * (float)mp[p] * (dx * dx + dy * dy + dz * dz) * (float)Hist_n/ (e_charge_mass * par->dt[p] * par->dt[p]*(float)Hist_max))<< endl;
      KEhist[p][index]++;
    }
    par->KEtot[p] = KE * 0.5 * mp[p] / (e_charge_mass * par->dt[p] * par->dt[p]) * r_part_spart; // as if these particles were actually samples of the greater thing
    par->nt[p] = nt*  r_part_spart;
   //   cout << p << " " << par->KEtot[p] << endl;
  }

  // Create a vtkPolyData object
  vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();

  // Add the FieldData to the PolyData
  vtkSmartPointer<vtkFieldData> fieldData = polyData->GetFieldData();
  vtkSmartPointer<vtkDoubleArray> timevalue = vtkSmartPointer<vtkDoubleArray>::New();
  timevalue->SetName("TimeValue");
  timevalue->InsertNextValue(t);
  fieldData->AddArray(timevalue);

  // Create a vtkPoints object to store the bin centers
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

  // Create a vtkDoubleArray object to store the bin counts
  vtkSmartPointer<vtkDoubleArray> ecounts = vtkSmartPointer<vtkDoubleArray>::New();
  ecounts->SetName("ecounts");
  // ecounts->SetNumberOfComponents(1);

  vtkSmartPointer<vtkDoubleArray> icounts = vtkSmartPointer<vtkDoubleArray>::New();
  icounts->SetName("icounts");
  // icounts->SetNumberOfComponents(1);

  // Fill the points array with data
  for (int i = 0; i < Hist_n; ++i)
  {
    double z = ((double)(i + 0.5) * (double)Hist_max) / (double)(Hist_n); // Calculate the center of the i-th bin
    points->InsertNextPoint(0.0, 0.0, z);                                 // Set the i-th point to the center of the i-th bin
    ecounts->InsertNextValue((double)(log(KEhist[0][i] + 1)));
    icounts->InsertNextValue((double)(log(KEhist[1][i] + 1)));
  }

  // Set the arrays as the data for the polyData object
  polyData->SetPoints(points);
  polyData->GetPointData()->AddArray(ecounts);
  polyData->GetPointData()->AddArray(icounts);

  // Write the polyData object to a file using VTK's XML file format
  vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
  writer->SetFileName((par->outpath + "KEhist_" + to_string(i_time) + ".vtp").c_str());
  writer->SetDataModeToBinary();
  writer->SetCompressorTypeToZLib(); // Enable compression
  writer->SetCompressionLevel(9);    // Set the level of compression (0-9)
  writer->SetInputData(polyData);
  writer->Write();
}

void save_hist1(int i_time, double t, int q[2][n_partd], float pos0x[2][n_partd], float pos0y[2][n_partd], float pos0z[2][n_partd], float pos1x[2][n_partd], float pos1y[2][n_partd], float pos1z[2][n_partd], par *par)
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

  // Create a polydata object
  vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();

  // Add the FieldData to the PolyData
  vtkSmartPointer<vtkFieldData> fieldData = polyData->GetFieldData();
  vtkSmartPointer<vtkDoubleArray> timevalue = vtkSmartPointer<vtkDoubleArray>::New();
  timevalue->SetName("TimeValue");
  timevalue->SetNumberOfTuples(1);
  timevalue->InsertNextValue(t);
  fieldData->AddArray(timevalue);

  long KEhist[2][Hist_n];
  memset(KEhist, 0, sizeof(KEhist));
  float coef[2];
  for (int p = 0; p < 2; ++p)
  {
    float KE = 0;
    int nt = 0;
    coef[p] = 0.5 * (float)mp[p] * (float)Hist_n / (e_charge_mass * par->dt[p] * par->dt[p] * (float)Hist_max);
    for (int i = 0; i < par->n_part[p]; ++i)
    {
      float dx = pos1x[p][i] - pos0x[p][i];
      float dy = pos1y[p][i] - pos0y[p][i];
      float dz = pos1z[p][i] - pos0z[p][i];
      float v2 = (dx * dx + dy * dy + dz * dz);
      unsigned int index = (int)floor(coef[p] * v2);
      KE += v2;
      nt += q[p][i];
      if (index < Hist_n)
        //        index = Hist_n - 1;
        //   if (index < 0)
        //     cout << "error index<0"<<(0.5 * (float)mp[p] * (dx * dx + dy * dy + dz * dz) * (float)Hist_n/ (e_charge_mass * par->dt[p] * par->dt[p]*(float)Hist_max))<< endl;
        KEhist[p][index]++;
    }
    par->KEtot[p] = KE * 0.5 * mp[p] / (e_charge_mass)*r_part_spart; // as if these particles were actually samples of the greater thing
    par->nt[p] = nt;
    cout << p << " " << par->KEtot[p] << endl;
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
  // table->AddColumn(timevalue);

  // Write the table to a file
  vtkSmartPointer<vtkDelimitedTextWriter> writer = vtkSmartPointer<vtkDelimitedTextWriter>::New();
  writer->SetFileName((par->outpath + "KEhist_" + to_string(i_time) + ".csv").c_str());
  writer->SetInputData(table);
  writer->Write();
}

/**
 * This corrects the order of dimensions for view in paraview, as opposed to save_vti which prints the raw data.
 */
void save_vti_c(string filename, int i,
                int ncomponents, double t,
                float data1[][n_space_divz][n_space_divy][n_space_divz], par *par)
{
  if (ncomponents > 3)
  {
    cout << "Error: Cannot write file " << filename << " - too many components" << endl;
    return;
  }
  int xi = (par->n_space_div[0] - 1) / maxcells + 1;
  int yj = (par->n_space_div[0] - 1) / maxcells + 1;
  int zk = (par->n_space_div[0] - 1) / maxcells + 1;
  int nx = par->n_space_div[0] / xi;
  int ny = par->n_space_div[1] / yj;
  int nz = par->n_space_div[2] / zk;

  vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New(); // Create the vtkImageData object
  imageData->SetDimensions(nx, ny, nz);                                           // Set the dimensions of the image data
  imageData->SetSpacing(par->dd[0] * xi, par->dd[1] * yj, par->dd[2] * zk);
  imageData->SetOrigin(par->posL[0], par->posL[1], par->posL[2]); // Set the origin of the image data
  imageData->AllocateScalars(VTK_FLOAT, ncomponents);
  imageData->GetPointData()->GetScalars()->SetName(filename.c_str());
  float *data2 = static_cast<float *>(imageData->GetScalarPointer()); // Get a pointer to the density field array

  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i)
        for (int c = 0; c < ncomponents; ++c)
          data2[(k * ny + j) * nx * ncomponents + i * ncomponents + c] = data1[c][k * zk][j * yj][i * xi];

  // Create a vtkDoubleArray to hold the field data
  vtkSmartPointer<vtkDoubleArray> timeArray = vtkSmartPointer<vtkDoubleArray>::New();
  timeArray->SetName("TimeValue");
  timeArray->SetNumberOfTuples(1);
  timeArray->SetValue(0, t);

  // Add the field data to the image data
  vtkSmartPointer<vtkFieldData> fieldData = imageData->GetFieldData();
  fieldData->AddArray(timeArray);

  vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New(); // Create the vtkXMLImageDataWriter object
  writer->SetFileName((par->outpath + filename + "_" + to_string(i) + ".vti").c_str());               // Set the output file name                                                                     // Set the time value
  writer->SetDataModeToBinary();
  // writer->SetCompressorTypeToLZ4();
  writer->SetCompressorTypeToZLib(); // Enable compression
  writer->SetCompressionLevel(9);    // Set the level of compression (0-9)
  writer->SetInputData(imageData);   // Set the input image data
                                     // Set the time step value
  writer->Write();                   // Write the output file
}

void save_vtp(string filename, int i, uint64_t num, double t, float data[n_output_part], float points1[n_output_part][3])
{
  // Create a polydata object
  vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
  // Add the FieldData to the PolyData
  vtkSmartPointer<vtkFieldData> fieldData = polyData->GetFieldData();
  vtkSmartPointer<vtkDoubleArray> timeArray = vtkSmartPointer<vtkDoubleArray>::New();
  timeArray->SetName("TimeValue");
  timeArray->SetNumberOfTuples(1);
  timeArray->SetValue(0, t);
  fieldData->AddArray(timeArray);

  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  vtkSmartPointer<vtkFloatArray> kineticEnergy = vtkSmartPointer<vtkFloatArray>::New();
  kineticEnergy->SetName("KE");
  for (int i = 0; i < num; ++i)
  {
    points->InsertNextPoint(points1[i][0], points1[i][1], points1[i][2]);
    kineticEnergy->InsertNextValue(data[i]);
  }

  polyData->SetPoints(points);
  polyData->GetPointData()->AddArray(kineticEnergy);
  // Write the output file
  vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
  writer->SetFileName((par->outpath + filename + "_" + to_string(i) + ".vtp").c_str());
  writer->SetDataModeToBinary();
  writer->SetCompressorTypeToZLib(); // Enable compression
  writer->SetCompressionLevel(9);    // Set the level of compression (0-9)
  writer->SetInputData(polyData);
  writer->Write();
}
