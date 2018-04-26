#pragma once
#include<iostream>
#include<omp.h>
#include<string>
#include<fstream>

using namespace std;

//***************************************************************************************************
//*****************************************sequential************************************************
//***************************************************************************************************


void sequential_k_means(float **data, int num_samples, int num_coordinates, int num_centroids)
{
	float **centroids = new float*[num_centroids];
	float **temp_centroids = new float*[num_centroids];

	//cout << "Centroids initialization" << endl;
	for (int i = 0; i < num_centroids; i++)
	{
		centroids[i] = new float[num_coordinates];
		centroids[i] = data[i];

		temp_centroids[i] = new float[num_coordinates];
		temp_centroids[i] = 0;

		//for (int j = 0; j < num_coordinates; j++)
		//cout << centroids[i][j] << "  ";

		//cout << endl << endl;
	}


	bool chang = false;
	bool thresh = true;

	int *belong_to = new int[num_samples];

	//loop untill 2 loops have same values 
	while (chang)
	{
		chang = false;
		thresh = true;

		//calculate from"centroids"	, put calculated centroids in"temp_centroids"

		for (int sample_indx = 0; sample_indx < num_samples; sample_indx++)
		{
			float sum = 5000;

			for (int centroid_indx = 0; centroid_indx < num_centroids; centroid_indx++)
			{
				float temp_sum = 0;

				for (int coordinate_indx = 0; coordinate_indx < num_coordinates; coordinate_indx++)
				{
					temp_sum += pow(data[sample_indx][coordinate_indx] - centroids[centroid_indx][coordinate_indx], 2);
				}

				temp_sum = sqrt(temp_sum);

				if (temp_sum < sum)
				{
					sum = temp_sum;
					belong_to[sample_indx] = centroid_indx;
				}
			}
		}

		//zeroooo temp_centroids to add in it to get average in it finally
		for (int i = 0; i < num_centroids; i++)
		{
			for (int j = 0; j < num_coordinates; j++)
			{
				temp_centroids[i][j] = 0;
			}
		}


		//get sum (new centroids) ,accomulate values on temp_centroids
		for (int data_indx = 0; data_indx < num_samples; data_indx++)
		{
			for (int coordinate_indx = 0; coordinate_indx < num_coordinates; coordinate_indx++)
			{
				temp_centroids[belong_to[data_indx]][coordinate_indx] += data[data_indx][coordinate_indx];
			}

		}

		//divide to get average in temp_centroids
		for (int temp_indx = 0; temp_indx < num_centroids; temp_indx++)
		{
			for (int coordinate_indx = 0; coordinate_indx < num_coordinates; coordinate_indx++)
			{
				temp_centroids[temp_indx][coordinate_indx] /= num_coordinates;
			}
		}


		//assign in centroids if centroids!=temp_centroids
		for (int i = 0; i < num_centroids; i++)
		{
			float sum = 0;

			for (int j = 0; j < num_coordinates; j++)
			{
				if (temp_centroids[i][j] != centroids[i][j])
				{
					sum += pow(temp_centroids[i][j] - centroids[i][j], 2);
					chang = true;
					centroids[i][j] = temp_centroids[i][j];
				}
			}

			if (sqrt(sum) > 0.001)
				thresh = false;
		}

		//if reach the threshold break
		if (thresh)
			break;
	}

	cout << "Sequential Result " << endl << endl;
	for (int i = 0; i < num_centroids; i++)
	{
		cout << "centroid [" << i << "]" << " :";
		for (int j = 0; j < num_coordinates; j++)
		{
			cout << centroids[i][j] << "   ";
		}
		cout << endl;
	}
	cout << endl << endl;
}

//****************************************************************************************************
//***********************************parallel*********************************************************
//****************************************************************************************************


float** parallel_k_means(float **data, int num_samples, int num_coordinates, int num_centroids)
{
	float **centroids = new float*[num_centroids];
	float **temp_centroids = new float*[num_centroids];

	//cout << "Centroids initialization" << endl;
	//**************************************************************************************parallel region
#pragma omp parallel for //iterations will be divided on 8 threads beacause of core i7 
	for (int i = 0; i < num_centroids; i++)
	{
		centroids[i] = new float[num_coordinates];
		centroids[i] = data[i];

		temp_centroids[i] = new float[num_coordinates];
		temp_centroids[i] = 0;

		//for (int j = 0; j < num_coordinates; j++)
		//cout << centroids[i][j] << "  ";

		//cout << endl << endl;
	}


	bool chang = false;
	bool thresh = true;

	int *belong_to = new int[num_samples];

	//loop untill 2 loops have same values 
	while (chang)
	{
		chang = false;
		thresh = true;

		//calculate from "centroids"	, put calculated centroids in "temp_centroids"

		for (int sample_indx = 0; sample_indx < num_samples; sample_indx++)
		{
			float sum = 5000;

			for (int centroid_indx = 0; centroid_indx < num_centroids; centroid_indx++)
			{
				float temp_sum = 0;

				//***********************************************************************************parallel region
#pragma omp parallel for reduction(+:temp_sum)
				for (int coordinate_indx = 0; coordinate_indx < num_coordinates; coordinate_indx++)
				{
					temp_sum += pow(data[sample_indx][coordinate_indx] - centroids[centroid_indx][coordinate_indx], 2);
				}

				temp_sum = sqrt(temp_sum);

				if (temp_sum < sum)
				{
					sum = temp_sum;
					belong_to[sample_indx] = centroid_indx;
				}
			}
		}

		//zeroooo temp_centroids to add in it to get average in it finally
		//******************************************************************************parallel region
#pragma omp parallel for //collapse(2) //collapse need version 3.0
		for (int i = 0; i < num_centroids; i++)
		{
			for (int j = 0; j < num_coordinates; j++)
			{
				temp_centroids[i][j] = 0;
			}
		}


		//get sum (new centroids) ,accomulate values on temp_centroids
		//******************************************************************************parallel region
#pragma omp parallel for //collapse(2) //collapse need version 3.0
		for (int data_indx = 0; data_indx < num_samples; data_indx++)
		{
			for (int coordinate_indx = 0; coordinate_indx < num_coordinates; coordinate_indx++)
			{
				temp_centroids[belong_to[data_indx]][coordinate_indx] += data[data_indx][coordinate_indx];
			}

		}

		//divide to get average in temp_centroids
		//******************************************************************************parallel region
#pragma omp parallel for //collapse(2) //collapse need version 3.0
		for (int temp_indx = 0; temp_indx < num_centroids; temp_indx++)
		{
			for (int coordinate_indx = 0; coordinate_indx < num_coordinates; coordinate_indx++)
			{
				temp_centroids[temp_indx][coordinate_indx] /= num_coordinates;
			}
		}


		//assign in centroids if centroids!=temp_centroids
		//******************************************************************************parallel region
#pragma omp parallel for
		for (int i = 0; i < num_centroids; i++)
		{
			float sum = 0;

			for (int j = 0; j < num_coordinates; j++)
			{
				if (temp_centroids[i][j] != centroids[i][j])
				{
					sum += pow(temp_centroids[i][j] - centroids[i][j], 2);
					chang = true;
					centroids[i][j] = temp_centroids[i][j];
				}
			}

			//******************************************************************************one thread at a time 
#pragma omp critical
			if (sqrt(sum) > 0.001)
				thresh = false;
		}

		//if reach the threshold break
		if (thresh)
			break;
	}

	/*

	cout << "Parralel Result " << endl << endl;
	for (int i = 0; i < num_centroids; i++)
	{
	cout << "centroid [" << i << "]" << " :";
	for (int j = 0; j < num_coordinates; j++)
	{
	cout << centroids[i][j] << "   ";
	}
	cout << endl;
	}
	cout << endl << endl;

	*/

	return centroids;
}

//***********************************************************************************
//*********************************main**********************************************
//***********************************************************************************


int main()
{
	//IrisDataset
	//******************************************************************************************
	//******************************************************************************************
	//reading data from txt 
	//******************************************************************************************
	//******************************************************************************************
	ifstream myfile("IrisDataset.txt");

	int n, k;
	string line;

	myfile >> n >> k;

	float **data = new float*[n];

	int indx = 0;

	while (!myfile.eof())
	{
		data[indx] = new float[4];
		string x;

		for (int i = 0; i < 3; i++)
		{
			getline(myfile, x, ',');
			data[indx][i] = atof(x.c_str());
		}

		getline(myfile, x, '\n');
		data[indx][3] = atof(x.c_str());

		indx++;
	}

	/*
	cout << "Data" << endl;
	for (int i = 0; i < n; i++)
	{
	for (int j = 0; j < 4; j++)
	{
	cout << data[i][j] << "     ";
	}
	cout << endl;
	}

	cout << endl << endl;
	*/

	//********************************************************************************************
	//********************************************************************************************
	//********************************************************************************************

	float **parallel_centroids = new float*[k];

	for (int i = 0; i < k; i++)
	{
		parallel_centroids[i] = new float[4];
	}

	//sequential function write it's result on console
	sequential_k_means(data, n, 4, k);

	parallel_centroids = parallel_k_means(data, n, 4, k);

	//write centroids of parallel function in a file
	//IrisDataset_cluster_centres
	ofstream outfile;
	outfile.open("IrisDataset_cluster_centres.txt");

	//write parallel function result on console
	cout << "Parralel Result " << endl << endl;
	for (int i = 0; i < k; i++)
	{
		outfile << "centroid [" << i << "]" << " :";
		cout << "centroid [" << i << "]" << " :";
		for (int j = 0; j < 4; j++)
		{
			outfile << parallel_centroids[i][j] << ",";
			cout << parallel_centroids[i][j] << "   ";
		}
		outfile << "\n";
		cout << endl;
	}
	cout << endl << endl;

	cout << "Centroids File is Created :D " << endl << endl;


	return 0;
}

