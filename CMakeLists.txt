cmake_minimum_required(VERSION 3.15...3.30)
project(blendcorpus_helpers LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(helpers blendcorpus/data/helpers.cpp)
target_compile_features(helpers PRIVATE cxx_std_11)

# Install into the blendcorpus/data/ subpackage so it is importable
# as ``from blendcorpus.data import helpers``.
install(TARGETS helpers LIBRARY DESTINATION blendcorpus/data)
