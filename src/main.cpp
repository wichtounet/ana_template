//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <unordered_map>

#include <stdio.h>
#include <dirent.h>

#include "dll/rbm.hpp"
#include "dll/dbn.hpp"
#include "dll/test.hpp"

#include "cpp_utils/data.hpp"

#include "io.hpp"

using sample_t = std::vector<float>;

void read_data(
    const std::string& pt_samples_file, const std::string& ft_samples_file, const std::string& ft_labels_file,
    std::vector<sample_t>& pt_samples, std::vector<sample_t>& ft_samples, std::vector<std::size_t>& ft_labels);

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

std::size_t count_distinct(std::vector<std::size_t> v){
    std::sort(v.begin(), v.end());
    return std::distance(v.begin(), std::unique(v.begin(), v.end()));
}

int main(int argc, char* argv[]){
    if(argc < 4){
        std::cout << "Not enough arguments" << std::endl;
    }

    std::string pt_samples_file(argv[1]);
    std::string ft_samples_file(argv[2]);
    std::string ft_labels_file(argv[3]);

    //1. Create the DBN

    auto dbn = std::make_unique<dbn_t>();

    //1.1 Configuration of the pretraining

    //dbn->layer<0>().learning_rate = 0.1;
    dbn->layer<0>().initial_momentum = 0.9;
    dbn->layer<0>().final_momentum = 0.9;
    //dbn->layer<1>().learning_rate = 0.1;
    dbn->layer<1>().initial_momentum = 0.9;
    dbn->layer<1>().final_momentum = 0.9;
    //...

    //1.2 Configuration of the fine-tuning

    dbn->learning_rate = 0.77;

    //2. Read dataset

    std::vector<sample_t> pt_samples;       //The pretraining samples
    std::vector<sample_t> ft_samples;       //The finetuning samples
    std::vector<std::size_t> ft_labels;     //The finetuning labels

    read_data(pt_samples_file, ft_samples_file, ft_labels_file, pt_samples, ft_samples, ft_labels);

    auto labels = count_distinct(ft_labels);  //Number of labels

    std::cout << "There are " << labels << " different labels" << std::endl;

    //3. Train the DBN layers for N epochs

    dbn->pretrain(pt_samples, 10);

    //4. Fine tune the DBN for M epochs

    auto ft_error = dbn->fine_tune(
        ft_samples, ft_labels,
        labels,                   //Number of labels
        50);                      //number of epochs

    std::cout << "Fine-tuning error: " << ft_error << std::endl;

    //5. Store the file if you want to save it for later

    dbn->store("file.dat"); //Store to file

    return 0;
}

std::string remove_extension(const std::string& file, const std::vector<std::string>& extensions){
    for(auto& extension : extensions){
        auto extension_length = extension.size();

        if(file.size() <= extension_length){
            continue;
        }

        if(std::string(file.begin() + file.size() - extension_length, file.end()) == extension){
            return std::string(file.begin(), file.begin() + file.size() - extension_length);
        }
    }

    return file;
}

void read_samples(const std::string& file, std::vector<sample_t>& samples);
void read_labels(const std::string& file, std::vector<std::size_t>& samples);

void read_data(
    const std::string& pt_samples_file, const std::string& ft_samples_file, const std::string& ft_labels_file,
    std::vector<sample_t>& pt_samples, std::vector<sample_t>& ft_samples, std::vector<std::size_t>& ft_labels){

    std::vector<std::string> feature_extension{"feat"};
    std::vector<std::string> label_extension{"framelab", "3phnlab"};

    //Extract the list of files from the description files
    auto pt_samples_files = ana::get_files(pt_samples_file, feature_extension);
    auto ft_samples_files = ana::get_files(ft_samples_file, feature_extension);
    auto ft_labels_files = ana::get_files(ft_labels_file, label_extension);

    for(auto& file : pt_samples_files){
        read_samples(file, pt_samples);
    }

    for(auto& s_file : ft_samples_files){
        bool found = false;

        for(auto& l_file : ft_labels_files){
            auto clean_s = remove_extension(s_file, feature_extension);
            auto clean_l = remove_extension(l_file, label_extension);

            if(clean_l == clean_s){
                read_samples(s_file, ft_samples);
                read_labels(l_file, ft_labels);

                found = true;
                break;
            }

            if(std::count(clean_s.begin(), clean_s.end(), '/') > 1 && std::count(clean_l.begin(), clean_l.end(), '/') > 1){
                auto last_s = clean_s.find_last_of('/');
                auto last_l = clean_l.find_last_of('/');

                auto prelast_s = clean_s.find_last_of('/', last_s - 1);
                auto prelast_l = clean_l.find_last_of('/', last_l - 1);

                auto clean_clean_s =
                        std::string(clean_s.begin(), clean_s.begin() + prelast_s + 1)
                    +   std::string(clean_s.begin() + last_s, clean_s.end());

                auto clean_clean_l =
                        std::string(clean_l.begin(), clean_l.begin() + prelast_l + 1)
                    +   std::string(clean_l.begin() + last_l, clean_l.end());

                if(clean_clean_l == clean_clean_s){
                    read_samples(s_file, ft_samples);
                    read_labels(l_file, ft_labels);

                    found = true;
                    break;
                }
            }
        }

        if(!found){
            std::cout << "No equivalent found for " << s_file << std::endl;
        }
    }

    std::cout << "A total of " << pt_samples.size() << " window samples were read for pretraining" << std::endl;
    std::cout << "A total of " << ft_samples.size() << " window samples were read for fine-tuning" << std::endl;
    std::cout << "A total of " << ft_labels.size() << " window labels were read for pretraining" << std::endl;
}

void read_samples(const std::string& file, std::vector<sample_t>& samples){
    std::cout << "Read samples from file \"" << file << "\"" << std::endl;

    std::vector<sample_t> raw_samples;

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
    }

    static constexpr const std::size_t Left = (N - 1) / 2;
    static constexpr const std::size_t Right = (N - 1) / 2;

    for(std::size_t i = Left; i < raw_samples.size() - Right; i += Stride){
        sample_t sample;

        for(std::size_t x = 0; x < N; ++x){
            for(auto& feature : raw_samples[i - Left + x]){
                sample.push_back(feature);
            }
        }

        samples.push_back(std::move(sample));
    }

    std::cout << samples.size() << " window samples were read" << std::endl;
}

std::unordered_map<std::string, std::size_t> mapper;

void read_labels(const std::string& file, std::vector<std::size_t>& labels){
    std::cout << "Read labels from file \"" << file << "\"" << std::endl;

    std::vector<std::size_t> raw_labels;

    std::ifstream infile(file);

    std::string line;
    while (std::getline(infile, line)){
        if(!mapper.count(line)){
            mapper[line] = mapper.size();
        }

        raw_labels.push_back(mapper[line]);
    }

    std::cout << raw_labels.size() << " raw labels were read" << std::endl;

    static constexpr const std::size_t Left = (N - 1) / 2;
    static constexpr const std::size_t Right = (N - 1) / 2;

    for(std::size_t i = Left; i < raw_labels.size() - Right; i += Stride){
        labels.push_back(raw_labels[i]);
    }

    std::cout << labels.size() << " window labels were read" << std::endl;
}
