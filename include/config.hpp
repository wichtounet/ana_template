//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ANA_TEMPLATE_CONFIG_HPP
#define ANA_TEMPLATE_CONFIG_HPP

static constexpr const std::size_t N = 11;
static constexpr const std::size_t Features = 43;
static constexpr const std::size_t Stride = 11;

//Putting lazy_pt = true uses lazy iterators, in which case, memory will be not be read at once
//at the beginning of the program but rather only when necessary
//This allow pretraining to work on arbitrary large dataset. However, this means that each data file
//will be read and normalized at least once each epoch. The overhead may be very large.
static constexpr const bool lazy_pt = false;

//Putting lazy_ft = true uses lazy iterators for fine-tuning, in which case, memory will be not be read at
//once at the beginning of the program but rather only when necessary
//This allow fine-tuning to work on arbitrary large dataset. However, this means that each data file
//will be read and normalized at least once each epoch. The overhead may be very large.
static constexpr const bool lazy_ft = false;

#endif
