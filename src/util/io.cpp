#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

static vector<vector<float>> readvec(string inname) {
    string filename = inname;
    int dim = 5;
    string line;
    vector<vector<float>> data;
    std::ifstream file(filename);
    while(getline(file, line)) {
        vector<float> onevec;
        for (int i = 0; i < dim; i++) {
            float a;
            file >> a;
            onevec.push_back(a);
        }
        data.push_back(onevec);
    }
    cout << "read done" << endl;
    return data;
}

static unsigned int writevec(vector<vector<float>> matrix, string outname) {
    ofstream output(outname);
    int dim = matrix.at(0).size();
    for(int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < dim; j++) {
            output << matrix.at(i).at(j) << " ";
        }
        output << "\n";
    }
    return 0;
}

static unsigned int writevec_str(vector<vector<string>> matrix, string outname) {
    ofstream output(outname);
    int dim = matrix.at(0).size();
    for(int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < dim; j++) {
            output << matrix.at(i).at(j) << "\t";
        }
        output << "\n";
    }
    return 0;
}