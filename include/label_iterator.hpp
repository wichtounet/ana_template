//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ANA_TEMPLATE_LABEL_ITERATOR_HPP
#define ANA_TEMPLATE_LABEL_ITERATOR_HPP

#include <vector>
#include <string>
#include <mutex>

#include "data.hpp"

namespace ana {

struct label_iterator : std::iterator<std::input_iterator_tag, ana::label_t> {
    const std::vector<std::string>& file_names;

    std::size_t current_file = 0;
    std::vector<ana::label_t> labels;
    std::size_t current_label = 0;

    static std::string cached;
    static std::vector<ana::label_t> cache;
    static std::mutex m;

    label_iterator(const std::vector<std::string>& file_names, std::size_t i = 0)
            : file_names(file_names), current_file(i) {
        if(current_file < file_names.size()){
            read_labels(file_names[current_file], labels);
        }
    }

    label_iterator(const label_iterator& rhs) = default;
    label_iterator& operator=(const label_iterator& rhs) = default;

    static void read_labels(const std::string& name, std::vector<ana::label_t>& labels){
        std::unique_lock<std::mutex> lock(m);

        if(!cache.empty() && cached == name){
            labels = cache;
        } else {
            cache.clear();
            ana::read_labels(name, cache);

            labels = cache;
            cached = name;
        }
    }

    bool operator==(const label_iterator& rhs){
        if(current_file == file_names.size() && current_file == rhs.current_file){
            return true;
        } else {
            return current_file == rhs.current_file && current_label == rhs.current_label;
        }
    }

    bool operator!=(const label_iterator& rhs){
        return !(*this == rhs);
    }

    ana::label_t& operator*(){
        return labels[current_label];
    }

    ana::label_t* operator->(){
        return &labels[current_label];
    }

    label_iterator operator++(){
        if(current_label == labels.size() - 1){
            ++current_file;
            current_label = 0;

            if(current_file < file_names.size()){
                read_labels(file_names[current_file], labels);
            }
        } else {
            ++current_label;
        }

        return *this;
    }

    label_iterator operator++(int){
        label_iterator it = *this;
        ++(*this);
        return it;
    }
};

} //end of namespace ana

#endif
