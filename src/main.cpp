//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.  // (See accompanying file LICENSE or copy at //  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include "dll/rbm.hpp"
#include "dll/dbn.hpp"
#include "dll/test.hpp"

#include "cpp_utils/data.hpp"

#include "config.hpp" //Edit this file to change the configuration
#include "io.hpp"
#include "data.hpp"
#include "sample_iterator.hpp"

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

namespace {

std::size_t count_distinct(std::vector<std::size_t> v){
    std::sort(v.begin(), v.end());
    return std::distance(v.begin(), std::unique(v.begin(), v.end()));
}

} //end of anonymous namespace

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

    std::vector<ana::sample_t> pt_samples;       //The pretraining samples
    std::vector<ana::sample_t> ft_samples;       //The finetuning samples
    std::vector<std::size_t> ft_labels;          //The finetuning labels

    ana::read_data(pt_samples_file, ft_samples_file, ft_labels_file, pt_samples, ft_samples, ft_labels, lazy_pt);

    auto labels = count_distinct(ft_labels);  //Number of labels

    std::cout << "There are " << labels << " different labels" << std::endl;

    //3. Train the DBN layers for N epochs

    if(lazy_pt){
        std::vector<std::string> feature_extension{"feat"};
        auto pt_samples_files = ana::get_files(pt_samples_file, feature_extension);

        ana::sample_iterator it(pt_samples_files);
        ana::sample_iterator end(pt_samples_files, pt_samples_files.size());

        dbn->pretrain(it, end, 10);
    } else {
        dbn->pretrain(pt_samples, 10);
    }

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
