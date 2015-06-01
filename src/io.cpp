//=======================================================================
// Copyright Baptiste Wicht 2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <vector>
#include <iostream>
#include <string>
#include <fstream>

#include <dirent.h>
#include <sys/stat.h>

#include "io.hpp"

using sample_t = std::vector<float>;

namespace {

bool ends_with(const std::string& file, const std::vector<std::string>& extensions){
    for(auto& extension : extensions){
        auto extension_length = extension.size();

        if(file.size() <= extension_length){
            continue;
        }

        if(std::string(file.begin() + file.size() - extension_length, file.end()) == extension){
            return true;
        }
    }

    return false;
}

void handle(const std::string& file, std::vector<std::string>& files, const std::string& line, const std::vector<std::string>& extension){
    struct stat buffer;

    if(stat(line.c_str(), &buffer) == 0){
        if(S_ISDIR(buffer.st_mode)){
            struct dirent* entry;
            DIR* dp = opendir(line.c_str());

            if(dp){
                while((entry = readdir(dp))){
                    if(std::string(entry->d_name) == "." || std::string(entry->d_name) == ".."){
                        continue;
                    }

                    handle(file, files, line + "/" + std::string(entry->d_name), extension);
                }
            } else {
                printf("error: 1: The file \"%s\" contains an invalid entry (\"%s\")\n", file.c_str(), line.c_str());
            }
        } else if(S_ISREG(buffer.st_mode)){
            if(ends_with(line, extension)){
                files.push_back(line);
            } else {
                printf("error: 2: The file \"%s\" contains an invalid entry (\"%s\")\n", file.c_str(), line.c_str());
            }
        } else {
            printf("error: 3: The file \"%s\" contains an invalid entry (\"%s\")\n", file.c_str(), line.c_str());
        }
    } else {
        printf("error: 4: The file \"%s\" contains an invalid entry (\"%s\")\n", file.c_str(), line.c_str());
    }
}

} //end of anonymous namespace

std::vector<std::string> ana::get_files(const std::string& file, const std::vector<std::string>& extension){
    std::vector<std::string> files;

    std::ifstream istream(file);

    std::string line;
    while(istream >> line){
        handle(file, files, line, extension);
    }

    return files;
}
