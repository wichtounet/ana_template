//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ANA_TEMPLATE_DATA_HPP
#define ANA_TEMPLATE_DATA_HPP

#include <vector>
#include <string>
#include <utility>
#include <unordered_map>

#include "etl/etl.hpp"

namespace ana {

using sample_t = etl::dyn_vector<float>;
using label_t = std::size_t;

using files_t = std::vector<std::string>;
using paired_files_t = std::pair<files_t, files_t>;

paired_files_t get_paired_files(const std::string& ft_samples_file, const std::string& ft_labels_file);

void read_data(
    const std::string& pt_samples_file, const paired_files_t& ft_files,
    std::vector<sample_t>& pt_samples, std::vector<sample_t>& ft_samples, std::vector<std::size_t>& ft_labels,
    bool lazy_pretraining = false, bool lazy_fine_tuning = false);

std::unordered_map<std::size_t, std::string> reverse_mapper();

void read_samples(const paired_files_t& files, const std::string& file, std::vector<ana::sample_t>& samples, bool pt);
void read_labels(const std::string& file, std::vector<std::size_t>& labels);

} //end of namespace ana

#endif
