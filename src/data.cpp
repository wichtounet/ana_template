//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <sstream>

#include "config.hpp"
#include "io.hpp"
#include "data.hpp"

namespace {

constexpr const bool verbose = false;

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

} //end of anonymous namespace

std::unordered_map<std::string, std::size_t> mapper;

void ana::read_labels(const std::string& file, std::vector<std::size_t>& labels){
    if(verbose){
        std::cout << "Read labels from file \"" << file << "\"" << std::endl;
    }

    std::vector<std::size_t> raw_labels;

    std::ifstream infile(file);

    std::string line;
    while (std::getline(infile, line)){
        if(!mapper.count(line)){
            mapper[line] = mapper.size();
        }

        raw_labels.push_back(mapper[line]);
    }

    if(verbose){
        std::cout << raw_labels.size() << " raw labels were read" << std::endl;
    }

    static constexpr const std::size_t Left = (N - 1) / 2;
    static constexpr const std::size_t Right = (N - 1) / 2;

    for(std::size_t i = Left; i < raw_labels.size() - Right; i += Stride){
        labels.push_back(raw_labels[i]);
    }

    if(verbose){
        std::cout << labels.size() << " window labels were read" << std::endl;
    }
}

void ana::read_samples(const std::string& file, std::vector<ana::sample_t>& samples){
    if(verbose){
        std::cout << "Read samples from file \"" << file << "\"" << std::endl;
    }

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

    if(verbose){
        std::cout << raw_samples.size() << " raw samples were read" << std::endl;
    }

    for(std::size_t i = 0; i < Features; ++i){
        // Compute the mean

        float mean = 0.0;
        for(auto& sample : raw_samples){
            mean += sample[i];
        }

        mean /= raw_samples.size();

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
        ana::sample_t sample(Features * N);

        std::size_t j = 0;
        for(std::size_t x = 0; x < N; ++x){
            for(auto& feature : raw_samples[i - Left + x]){
                sample[j++] = feature;
            }
        }

        samples.push_back(std::move(sample));
    }

    if(verbose){
        std::cout << samples.size() << " window samples were read" << std::endl;
    }
}

std::pair<std::vector<std::string>, std::vector<std::string>> ana::get_paired_files(const std::string& ft_samples_file, const std::string& ft_labels_file){
    std::vector<std::string> feature_extension{"feat"};
    std::vector<std::string> label_extension{"framelab", "3phnlab"};

    //Extract the list of files from the description files
    auto ft_samples_files = ana::get_files(ft_samples_file, feature_extension);
    auto ft_labels_files = ana::get_files(ft_labels_file, label_extension);

    std::vector<std::string> samples_files;
    std::vector<std::string> labels_files;

    for(auto& s_file : ft_samples_files){
        bool found = false;

        for(auto& l_file : ft_labels_files){
            auto clean_s = remove_extension(s_file, feature_extension);
            auto clean_l = remove_extension(l_file, label_extension);

            if(clean_l == clean_s){
                samples_files.push_back(s_file);
                labels_files.push_back(l_file);

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
                    samples_files.push_back(s_file);
                    labels_files.push_back(l_file);

                    found = true;
                    break;
                }
            }
        }

        if(!found){
            std::cout << "No equivalent found for " << s_file << std::endl;
        }
    }

    return {samples_files, labels_files};
}

void ana::read_data(
    const std::string& pt_samples_file, const std::string& ft_samples_file, const std::string& ft_labels_file,
    std::vector<sample_t>& pt_samples, std::vector<sample_t>& ft_samples, std::vector<std::size_t>& ft_labels,
    bool lazy_pretraining, bool lazy_fine_tuning){

    std::vector<std::string> feature_extension{"feat"};
    std::vector<std::string> label_extension{"framelab", "3phnlab"};

    //If not lazy, read the pretraining files
    if(!lazy_pretraining){
        auto pt_samples_files = ana::get_files(pt_samples_file, feature_extension);

        for(auto& file : pt_samples_files){
            read_samples(file, pt_samples);
        }
    }

    //If not lazy, read the fine-tuning files
    if(!lazy_fine_tuning){
        std::vector<std::string> samples_files;
        std::vector<std::string> labels_files;

        std::tie(samples_files, labels_files) = get_paired_files(ft_samples_file, ft_labels_file);

        for(auto& samples_file : samples_files){
            read_samples(samples_file, ft_samples);
        }

        for(auto& labels_file : labels_files){
            read_labels(labels_file, ft_labels);
        }
    }

    std::cout << "A total of " << pt_samples.size() << " window samples were read for pretraining" << std::endl;
    std::cout << "A total of " << ft_samples.size() << " window samples were read for fine-tuning" << std::endl;
    std::cout << "A total of " << ft_labels.size() << " window labels were read for pretraining" << std::endl;
}
