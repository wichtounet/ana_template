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

#include "etl/etl.hpp"

namespace ana {

using sample_t = etl::dyn_vector<float>;

void read_data(
    const std::string& pt_samples_file, const std::string& ft_samples_file, const std::string& ft_labels_file,
    std::vector<sample_t>& pt_samples, std::vector<sample_t>& ft_samples, std::vector<std::size_t>& ft_labels,
    bool lazy_pretraining = false);

void read_samples(const std::string& file, std::vector<ana::sample_t>& samples);

} //end of namespace ana

#endif
