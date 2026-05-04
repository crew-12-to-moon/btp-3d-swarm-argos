cmake_minimum_required(VERSION 3.10)
project(btp_3d_fresh)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================
# ARGoS paths
# =========================
set(ARGOS_ROOT  /Users/harshvardhanmohite/argos3)
set(ARGOS_SRC   ${ARGOS_ROOT}/src)
set(ARGOS_BUILD ${ARGOS_ROOT}/build)

# =========================
# Lua paths
# =========================
find_path(LUA_INCLUDE_DIR
  NAMES lua.h
  PATHS
    /opt/homebrew/include
    /opt/homebrew/include/lua
    /opt/homebrew/include/lua5.5
    /opt/homebrew/include/lua5.4
    /opt/homebrew/include/lua5.3
    /opt/homebrew/opt/lua/include
    /opt/homebrew/opt/lua/include/lua
    /opt/homebrew/Cellar/lua
  PATH_SUFFIXES
    include
    include/lua
    include/lua5.5
    include/lua5.4
    include/lua5.3
)

find_library(LUA_LIBRARY
  NAMES lua lua5.5 lua5.4 lua5.3
  PATHS
    /opt/homebrew/lib
    /opt/homebrew/opt/lua/lib
)

if(NOT LUA_INCLUDE_DIR)
  message(FATAL_ERROR "lua.h not found. Run: find /opt/homebrew -name lua.h")
endif()

if(NOT LUA_LIBRARY)
  message(FATAL_ERROR "Lua library not found.")
endif()

message(STATUS "Lua include: ${LUA_INCLUDE_DIR}")
message(STATUS "Lua library: ${LUA_LIBRARY}")

# =========================
# ARGoS generated config.h shim
# ARGoS headers include <argos3/core/config.h>
# Your generated config is usually at:
# ~/argos3/build/core/config.h
# =========================
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/argos3/core)

configure_file(
  ${ARGOS_BUILD}/core/config.h
  ${CMAKE_BINARY_DIR}/argos3/core/config.h
  COPYONLY
)

# =========================
# Library search paths
# =========================
link_directories(
  ${ARGOS_BUILD}/core
  ${ARGOS_BUILD}/plugins/simulator/entities
  ${ARGOS_BUILD}/plugins/robots/prototype
)

# =========================
# Controller library
# =========================
add_library(btp_controller SHARED
  src/influential_controller.cpp
  src/decoy_controller.cpp
)

target_include_directories(btp_controller PRIVATE
  ${CMAKE_BINARY_DIR}
  ${ARGOS_SRC}
  ${ARGOS_BUILD}
  ${LUA_INCLUDE_DIR}
)

target_link_libraries(btp_controller
  argos3core_simulator
  argos3plugin_simulator_entities
  argos3plugin_simulator_prototype
  ${LUA_LIBRARY}
)

# =========================
# Loop functions library
# =========================
add_library(btp_loop SHARED
  src/btp_loop_functions.cpp
)

target_include_directories(btp_loop PRIVATE
  ${CMAKE_BINARY_DIR}
  ${ARGOS_SRC}
  ${ARGOS_BUILD}
  ${LUA_INCLUDE_DIR}
)

target_link_libraries(btp_loop
  argos3core_simulator
  argos3plugin_simulator_entities
  argos3plugin_simulator_prototype
  ${LUA_LIBRARY}
)
