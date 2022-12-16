#include "writer.h"


void write_vector_float_as_text(std::string outfile, std::vector<std::vector<float>> &data)
{
    std::ofstream wf(outfile.c_str(), std::ios::out);
    if(!wf) {
        std::cout << "Cannot open file!" << std::endl;
    }
    for(int j = 0; j < data.size(); j++){
        std::vector<float> temp = data[j];
        for(int i = 0; i < temp.size(); i++){
            wf << temp[i] << " ";
        }
        wf << "\n";
        
    }
    // wf << "\n";
    wf.close();
    if(!wf.good()) {
        std::cout << "Error occurred at writing time!" << std::endl;
    }

}