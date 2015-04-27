#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>

#include "dll/rbm.hpp"
#include "dll/dbn.hpp"
#include "dll/test.hpp"

#include "cpp_utils/data.hpp"

void read_data(const std::string& file, std::vector<std::vector<float>>& samples, std::vector<std::size_t>& labels);

static constexpr const std::size_t N = 11;
static constexpr const std::size_t Features = 42;
static constexpr const std::size_t Stride = 11;

//0. Configure the DBN

using dbn_t = dll::dbn_desc<dll::dbn_layers<
        //First RBM
          dll::rbm_desc<
              Features * N
            , 500               //Number of hidden units
            , dll::momentum
            , dll::batch_size<25>
            , dll::parallel // Comment this line to use only 1 thread
            , dll::init_weights
            , dll::weight_decay<>
            , dll::visible<dll::unit_type::GAUSSIAN>
        >::rbm_t
        //Second RBM
        , dll::rbm_desc<
            500, 200,
            dll::momentum,
            dll::batch_size<25>
        >::rbm_t
        //Third RBM
        , dll::rbm_desc<
            200, 10,
            dll::momentum,
            dll::batch_size<25>,
            dll::hidden<dll::unit_type::SOFTMAX>
        >::rbm_t>
    >::dbn_t;

int main(int argc, char* argv[]){
    if(argc < 2){
        std::cout << "Not enough arguments" << std::endl;
    }

    std::string file(argv[1]);

    //1. Create the DBN

    auto dbn = std::make_unique<dbn_t>();

    //1.1 Configuration

    //dbn->layer<0>().learning_rate = 0.1;
    //dbn->layer<0>().momentum = 0.1;
    //dbn->layer<1>().learning_rate = 0.1;
    //dbn->layer<1>().momentum = 0.1;
    //...

    //2. Read dataset

    std::vector<std::vector<float>> samples;     //All the samples
    std::vector<std::size_t> labels;              //All the labels

    read_data(file, samples, labels);

    //3. Train the DBN layers for 100 epochs

    dbn->pretrain(samples, 10);

    //4. Fine tune the DBN

    //auto ft_error = dbn->fine_tune(
        //samples, labels,
        //10,  //Number of labels
        //50); //number of epochs

    //5. Store the file if you want to save it for later

    dbn->store("file.dat"); //Store to file

    return 0;
}

void read_data(const std::string& file, std::vector<std::vector<float>>& samples, std::vector<std::size_t>&){
    std::cout << "Read data from file \"" << file << "\"" << std::endl;

    std::vector<std::vector<float>> raw_samples;

    std::ifstream infile(file);

    std::string line;
    while (std::getline(infile, line)){
        std::vector<float> sample;

        std::istringstream iss(line);
        float feature;

        while(iss >> feature){
            sample.push_back(feature);
        }

        raw_samples.push_back(std::move(sample));
    }

    std::cout << raw_samples.size() << " raw samples were read" << std::endl;

    for(std::size_t i = 0; i < Features; ++i){
        // Compute the mean

        float mean = 0.0;
        for(auto& sample : raw_samples){
            mean += sample[i];
        }

        //Normalize to zero-mean

        for(auto& sample : raw_samples){
            sample[i] -= mean;
        }

        //Compute the variance

        float std = 0.0;
        for(auto& sample : raw_samples){
            std += sample[i] * sample[i];
        }

        std = std::sqrt(std / raw_samples.size());

        //Normalize to unit variance

        if(std != 0.0){
            for(auto& sample : raw_samples){
                sample[i] /= std;
            }
        }

        if(i == 20){
            for(auto& feature : raw_samples[i]){
                std::cout << feature << ",";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "Each feature was normalized to zero-mean and unit-variance" << std::endl;

    static constexpr const std::size_t Left = (N - 1) / 2;
    static constexpr const std::size_t Right = (N - 1) / 2;

    for(std::size_t i = Left; i < raw_samples.size() - Right; i += Stride){
        std::vector<float> sample;

        for(std::size_t x = 0; x < N; ++x){
            for(auto& feature : raw_samples[i - Left + x]){
                sample.push_back(feature);
            }
        }

        samples.push_back(std::move(sample));
    }

    std::cout << samples.size() << " window samples were read" << std::endl;

    //TODO Read labels
}
