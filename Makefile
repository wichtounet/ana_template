default: release

.PHONY: default release debug all clean

CXX=g++-4.9
LD=g++-4.9

include make-utils/flags.mk
include make-utils/cpp-utils.mk

CXX_FLAGS += -Idll/include -Idll/etl/include -Idll/etl/lib/include -Imnist/include -std=c++1y
LD_FLAGS += -lopencv_core -lopencv_imgproc -lopencv_highgui -ljpeg -lpthread

$(eval $(call auto_folder_compile,src))
$(eval $(call auto_add_executable,main))

release: release/bin/main
release_debug: release_debug/bin/main
debug: debug/bin/main

all: release release_debug debug

clean: base_clean

run: release_debug
	./release_debug/bin/main

include make-utils/cpp-utils-finalize.mk
