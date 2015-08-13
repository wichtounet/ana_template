//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.  // (See accompanying file LICENSE or copy at //  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include "dll/rbm.hpp"
#include "dll/dbn.hpp"
#include "dll/test.hpp"
#include "dll/dense_stochastic_gradient_descent.hpp"

#include "cpp_utils/data.hpp"

#include "config.hpp" //Edit this file to change the configuration
#include "io.hpp"
#include "data.hpp"
#include "sample_iterator.hpp"
#include "label_iterator.hpp"

//0. Configure the DBN

using dbn_t = dll::dbn_desc<dll::dbn_layers<
        //First RBM
          dll::rbm_desc<
              Features * N              // Number of input features
            , 500                       // Number of hidden units
            , dll::momentum
            , dll::batch_size<50>
            , dll::weight_decay<>
            , dll::visible<dll::unit_type::GAUSSIAN>
        >::rbm_t
        //Second RBM
        , dll::rbm_desc<
            500, 200
            , dll::momentum             // Use momentum during training
            , dll::batch_size<50>
        >::rbm_t
        //Third RBM
        , dll::rbm_desc<
            200
            , 42                        //This is the number of labels
            , dll::momentum
            , dll::batch_size<50>
            , dll::hidden<dll::unit_type::SOFTMAX>
        >::rbm_t>,
      dll::memory                               //Reduce memory consumption of the DBN (by using lazy iterators)
    , dll::big_batch_size<100>                  //Save some file readings
    , dll::batch_size<100>                      //Mini-batch for SGD
    , dll::trainer<dll::dense_sgd_trainer>      //Use SGD in place of CG
    , dll::momentum                             //Use momentum for SGD
    , dll::weight_decay<>                       //Use L2 weight decay for SGD
    >::dbn_t;

namespace ana {

template<typename DBN>
void test(DBN& dbn, paired_files_t& paired_files, std::vector<sample_t>& ft_samples, std::vector<std::size_t>& ft_labels);

std::size_t count_distinct(std::vector<std::size_t> v){
    std::sort(v.begin(), v.end());
    return std::distance(v.begin(), std::unique(v.begin(), v.end()));
}

template<typename DBN>
void generate_features(DBN& dbn, const std::string& pt_samples_file, const std::string& ft_samples_file, const std::string& ft_labels_file);

} //end of ana namespace

int main(int argc, char* argv[]){
    if(argc < 5){
        std::cout << "Not enough arguments" << std::endl;
        return 1;
    }

    std::string action(argv[1]);
    std::string pt_samples_file(argv[2]);
    std::string ft_samples_file(argv[3]);
    std::string ft_labels_file(argv[4]);

    if(!(action == "train" || action == "feat" || action == "test" || action == "train_feat" || action == "train_test")){
        std::cout << "Invalid action :" << action << std::endl;
        return 2;
    }

    //1. Create the DBN

    auto dbn = std::make_unique<dbn_t>();

    //1.1 Configuration of the pretraining

    //dbn->layer_get<0>().learning_rate *= 0.1;
    //dbn->layer_get<0>().initial_momentum = 0.9;
    //dbn->layer_get<0>().final_momentum = 0.9;

    dbn->layer_get<1>().learning_rate = 0.05;
    dbn->layer_get<1>().initial_momentum = 0.9;
    dbn->layer_get<1>().final_momentum = 0.9;
    //...

    //1.2 Configuration of the fine-tuning

    dbn->learning_rate = 0.1;
    dbn->initial_momentum = 0.9;

    //2. Read dataset

    //Collect the paired files
    auto paired_files = ana::get_paired_files(ft_samples_file, ft_labels_file);

    if(action == "train" || action == "train_feat" || action == "train_test"){
        //Collection of files for pretraining
        std::vector<std::string> feature_extension{"feat"};
        auto pt_samples_files = ana::get_files(pt_samples_file, feature_extension);

        std::vector<ana::sample_t> pt_samples;       //The pretraining samples
        std::vector<ana::sample_t> ft_samples;       //The finetuning samples
        std::vector<std::size_t> ft_labels;          //The finetuning labels

        ana::read_data(pt_samples_file, paired_files, pt_samples, ft_samples, ft_labels, lazy_pt, lazy_ft);

        std::cout << "There are " << ana::count_distinct(ft_labels) << " different labels" << std::endl;

        //3. Train the DBN layers for N epochs

        std::size_t pt_epochs = 10;

        if(lazy_pt){
            ana::sample_iterator it(paired_files, pt_samples_files, true);
            ana::sample_iterator end(paired_files, pt_samples_files, true, pt_samples_files.size());

            dbn->pretrain(it, end, pt_epochs);
        } else {
            dbn->pretrain(pt_samples, pt_epochs);
        }

        //4. Fine tune the DBN for M epochs

        std::size_t ft_epochs = 20;

        if(lazy_ft){
            ana::sample_iterator it(paired_files, pt_samples_files, false);
            ana::sample_iterator end(paired_files, pt_samples_files, false, paired_files.first.size());

            ana::label_iterator lit(paired_files);
            ana::label_iterator lend(paired_files, paired_files.first.size());

            auto ft_error = dbn->fine_tune(it, end, lit, lend, ft_epochs);

            std::cout << "Fine-tuning error: " << ft_error << std::endl;
        } else {
            auto ft_error = dbn->fine_tune(ft_samples, ft_labels, ft_epochs);

            std::cout << "Fine-tuning error: " << ft_error << std::endl;
        }

        //5. Store the file if you want to save it for later

        dbn->store("file.dat"); //Store to file

        if(action == "train_feat"){
            std::cout << "Generate features" << std::endl;
            ana::generate_features(*dbn, pt_samples_file, ft_samples_file, ft_labels_file);
        } else if(action == "train_test"){
            ana::test(*dbn, paired_files, ft_samples, ft_labels);
        }
    } else if(action == "feat"){
        dbn->load("file.dat"); //Load from file

        std::cout << "Generate features" << std::endl;
        ana::generate_features(*dbn, pt_samples_file, ft_samples_file, ft_labels_file);
    } else if(action == "test"){
        dbn->load("file.dat"); //Load from file

        std::vector<ana::sample_t> pt_samples;       //The pretraining samples
        std::vector<ana::sample_t> ft_samples;       //The finetuning samples
        std::vector<std::size_t> ft_labels;          //The finetuning labels

        ana::read_data(pt_samples_file, paired_files, pt_samples, ft_samples, ft_labels, true, lazy_ft);

        ana::test(*dbn, paired_files, ft_samples, ft_labels);
    }

    return 0;
}

namespace ana {

template<typename DBN>
void test(DBN& dbn, paired_files_t& paired_files, std::vector<sample_t>& ft_samples, std::vector<std::size_t>& ft_labels){
    std::cout << "\nTest\n";

    std::vector<std::string> pt_samples_files;

    std::vector<std::size_t> errors;
    std::size_t errors_tot = 0;
    std::size_t total = 0;

    auto rmap = ana::reverse_mapper();

    if(lazy_ft){
        ana::sample_iterator it(paired_files, pt_samples_files, false);
        ana::sample_iterator end(paired_files, pt_samples_files, false, paired_files.first.size());

        ana::label_iterator lit(paired_files);

        while(it != end){
            auto& samples = *it;
            auto& label = *lit;

            auto p = dbn.predict(samples);

            if(p != label){
                if(label >= errors.size()){
                    errors.resize(label+1);
                }

                ++errors[label];
                ++errors_tot;
            }

            ++total;

            ++it;
            ++lit;
        }
    } else {
        total = ft_samples.size();

        auto labels = ana::count_distinct(ft_labels);
        errors.resize(labels);

        for(std::size_t i = 0; i < ft_samples.size(); ++i){
            auto& samples = ft_samples[i];
            auto& label = ft_labels[i];

            auto p = dbn.predict(samples);

            if(p != label){
                ++errors[label];
                ++errors_tot;
            //std::cout << p << ":" << label << std::endl;
            }
        }
    }

    std::cout << "Accuracy: " << (total - errors_tot) / double(total) << std::endl;;
    std::cout << "Errors: " << errors_tot << std::endl;;

    for(std::size_t i = 0; i < errors.size(); ++i){
        std::cout << rmap[i] << " " << errors[i] << std::endl;
    }
}

template<std::size_t I, typename DBN, cpp_enable_if((I == DBN::layers))>
void generate_features_layer(DBN&, const std::vector<ana::sample_t>&, const std::string&){
    //Cool
}

template<std::size_t I, typename DBN, cpp_enable_if((I < DBN::layers))>
void generate_features_layer(DBN& dbn, const std::vector<ana::sample_t>& samples, const std::string& file){
    std::string target_file = std::string(file.begin(), file.end() - 4) + std::to_string(I) + ".bnf";

    std::ofstream out(target_file);

    for(auto& sample : samples){
        auto features = dbn.template activation_probabilities_sub<I>(sample);

        for(auto& feature : features){
            out <<  feature << ",";
        }

        out << '\n';
    }

    std::cout << '.';
    std::cout.flush();

    //Generate features for the next layer
    generate_features_layer<I+1>(dbn, samples, file);
}

template<typename DBN>
void generate_features(DBN& dbn, const std::string& pt_samples_file, const std::string& ft_samples_file, const std::string& ft_labels_file){
    std::vector<std::string> feature_extension{"feat"};
    std::vector<std::string> label_extension{"framelab", "3phnlab"};

    auto pt_samples_files = ana::get_files(pt_samples_file, feature_extension);

    auto paired_files = ana::get_paired_files(ft_samples_file, ft_labels_file);

    for(auto& file : pt_samples_files){
        std::vector<ana::sample_t> samples;
        ana::read_samples(paired_files, file, samples, true);

        generate_features_layer<0>(dbn, samples, file);
    }

    for(auto& file : paired_files.first){
        std::vector<ana::sample_t> samples;
        ana::read_samples(paired_files, file, samples, true);

        generate_features_layer<0>(dbn, samples, file);
    }
}

} //end of ana namespace
