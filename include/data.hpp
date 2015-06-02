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

#include "etl/etl.hpp"

namespace ana {

using sample_t = etl::dyn_vector<float>;
using label_t = std::size_t;

void read_data(
    const std::string& pt_samples_file, const std::string& ft_samples_file, const std::string& ft_labels_file,
    std::vector<sample_t>& pt_samples, std::vector<sample_t>& ft_samples, std::vector<std::size_t>& ft_labels,
    bool lazy_pretraining = false, bool lazy_fine_tuning = false);

std::pair<std::vector<std::string>, std::vector<std::string>> get_paired_files(const std::string& ft_samples_file, const std::string& ft_labels_file);

void read_samples(const std::string& file, std::vector<ana::sample_t>& samples);
void read_labels(const std::string& file, std::vector<std::size_t>& labels);

} //end of namespace ana

#endif
