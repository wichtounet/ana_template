//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ANA_TEMPLATE_IO_HPP
#define ANA_TEMPLATE_IO_HPP

#include <vector>
#include <string>

namespace ana {

std::vector<std::string> get_files(const std::string& file, const std::vector<std::string>& extension);

} //end of namespace ana

#endif
