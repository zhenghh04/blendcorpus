CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color
CPPFLAGS += $(shell python3 -m pybind11 --includes)
LIBNAME = helpers
# LIBEXT=.cpython-312-x86_64-linux-gnu.so
# LIBEXT = $(shell python3-config --extension-suffix)
LIBEXT = $(shell python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

default: $(LIBNAME)$(LIBEXT)

%$(LIBEXT): %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
