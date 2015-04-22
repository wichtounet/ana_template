#include <vector>
#include <iostream>

#include "dll/rbm.hpp"
#include "dll/dbn.hpp"
#include "dll/test.hpp"

#include "cpp_utils/data.hpp"

void read_data(std::vector<std::vector<double>>& samples, std::vector<std::size_t>& labels);

//0. Configure the DBN

using dbn_t = dll::dbn_desc<dll::dbn_layers<
        //First RBM
          dll::rbm_desc<
            28 * 28, 100,
            dll::momentum,
            dll::batch_size<25>,
            dll::init_weights
        >::rbm_t
        //Second RBM
        , dll::rbm_desc<
            100, 200,
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

int main(){
    //1. Create the DBN

    auto dbn = std::make_unique<dbn_t>();

    //2. Read dataset

    std::vector<std::vector<double>> samples;     //All the samples
    std::vector<std::size_t> labels;              //All the labels

    read_data(samples, labels);

    //3. Train the DBN layers for 100 epochs

    dbn->pretrain(samples, 10);

    //4. Train the SVM

    auto ft_error = dbn->fine_tune(
        samples, labels,
        10,  //Number of labels
        50); //number of epochs

    //5. Store the file if you want to save it for later

    dbn->store("file.dat"); //Store to file

    return 0;
}

void read_data(std::vector<std::vector<double>>& samples, std::vector<std::size_t>& labels){
    //TODO Read samples
    //TODO Read labels

    //cpp::normalize_each(samples); //For gaussian visible units
}
