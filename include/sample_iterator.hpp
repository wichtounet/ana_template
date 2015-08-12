//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ANA_TEMPLATE_SAMPLE_ITERATOR_HPP
#define ANA_TEMPLATE_SAMPLE_ITERATOR_HPP

#include <vector>
#include <string>
#include <mutex>

#include "data.hpp"

namespace ana {

struct sample_iterator : std::iterator<std::input_iterator_tag, ana::sample_t> {
    const ana::paired_files_t& file_names;
    const bool pt;

    std::size_t current_file = 0;
    std::vector<ana::sample_t> samples;
    std::size_t current_sample = 0;

    static std::string cached;
    static std::vector<ana::sample_t> cache;
    static std::mutex m;

    sample_iterator(const ana::paired_files_t& file_names, bool pt, std::size_t i = 0)
            : file_names(file_names), pt(pt), current_file(i) {
        if(current_file < file_names.first.size()){
            read_samples(pt, file_names, file_names.first[current_file], samples);
        }
    }

    sample_iterator(const sample_iterator& rhs) = default;
    sample_iterator& operator=(const sample_iterator& rhs) = default;

    static void read_samples(bool pt, const ana::paired_files_t& file_names, const std::string& name, std::vector<ana::sample_t>& samples){
        std::unique_lock<std::mutex> lock(m);

        if(!cache.empty() && cached == name){
            samples = cache;
        } else {
            cache.clear();
            ana::read_samples(file_names, name, cache, pt);

            samples = cache;
            cached = name;
        }
    }

    bool operator==(const sample_iterator& rhs){
        if(current_file == file_names.first.size() && current_file == rhs.current_file){
            return true;
        } else {
            return current_file == rhs.current_file && current_sample == rhs.current_sample;
        }
    }

    bool operator!=(const sample_iterator& rhs){
        return !(*this == rhs);
    }

    ana::sample_t& operator*(){
        return samples[current_sample];
    }

    ana::sample_t* operator->(){
        return &samples[current_sample];
    }

    sample_iterator operator++(){
        if(current_sample == samples.size() - 1){
            ++current_file;
            current_sample = 0;

            if(current_file < file_names.first.size()){
                read_samples(pt, file_names, file_names.first[current_file], samples);
            }
        } else {
            ++current_sample;
        }

        return *this;
    }

    sample_iterator operator++(int){
        sample_iterator it = *this;
        ++(*this);
        return it;
    }
};

} //end of namespace ana

#endif
