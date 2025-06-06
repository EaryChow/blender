# SPDX-FileCopyrightText: 2017-2023 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

if(WIN32)
  option(ENABLE_MSYS2 "Enable building of ffmpeg/libsndfile/fftw3/gmp by installing msys2" ON)
  option(MSYS2_USE_UPSTREAM_PACKAGES "Use upstream packages to bootstrap msys2, when OFF the blender mirror will be used" OFF)
endif()
option(FORCE_CHECK_HASH "Force a check of all hashses during CMake the configure phase" OFF)

cmake_host_system_information(RESULT NUM_CORES QUERY NUMBER_OF_LOGICAL_CORES)
set(MAKE_THREADS ${NUM_CORES} CACHE STRING "Number of threads to run make with")

# Any python module building with setup.py cannot use multiple theads on windows
# as they will try to write to the same .pdb file simultaniously which causes
# build errors.
if(WIN32)
  set(PYTHON_MAKE_THREADS 1)
else()
  set(PYTHON_MAKE_THREADS ${MAKE_THREADS})
endif()

if(NOT BUILD_MODE)
  set(BUILD_MODE "Release")
  message(STATUS "Build type not specified: defaulting to a release build.")
endif()
message(STATUS "BuildMode = ${BUILD_MODE}")

if(BUILD_MODE STREQUAL "Debug")
  set(LIBDIR ${CMAKE_CURRENT_BINARY_DIR}/Debug)
  set(MESON_BUILD_TYPE -Dbuildtype=debug)
else()
  set(LIBDIR ${CMAKE_CURRENT_BINARY_DIR}/Release)
  set(MESON_BUILD_TYPE -Dbuildtype=release)
endif()

set(DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/downloads" CACHE STRING "Path for downloaded files")

set(PACKAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/packages" CACHE PATH "default path for downloaded packages")
option(PACKAGE_USE_UPSTREAM_SOURCES "Use sources upstream to download the package sources, when OFF the blender mirror will be used" ON)

file(TO_CMAKE_PATH ${DOWNLOAD_DIR} DOWNLOAD_DIR)
file(TO_CMAKE_PATH ${PACKAGE_DIR} PACKAGE_DIR)
set(PATCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/patches)
set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/build)

message(STATUS "LIBDIR = ${LIBDIR}")
message(STATUS "DOWNLOAD_DIR = ${DOWNLOAD_DIR}")
message(STATUS "PACKAGE_DIR = ${PACKAGE_DIR}")
message(STATUS "PATCH_DIR = ${PATCH_DIR}")
message(STATUS "BUILD_DIR = ${BUILD_DIR}")

if(WIN32)
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "ARM64")
    set(BLENDER_PLATFORM_ARM ON)
    set(BLENDER_PLATFORM_WINDOWS_ARM ON)
  endif()
  set(PATCH_CMD ${DOWNLOAD_DIR}/msys2/msys64/usr/bin/patch.exe)
  set(LIBEXT ".lib")
  set(SHAREDLIBEXT ".lib")
  set(LIBPREFIX "")
  set(MESON ${LIBDIR}/python/Scripts/meson)
  # For OIIO and OSL
  set(COMMON_DEFINES /DPSAPI_VERSION=2 /DTINYFORMAT_ALLOW_WCHAR_STRINGS)

  if(MSVC_VERSION GREATER 1909)
    # Some deps with warnings as error aren't quite ready for dealing with the new 2017 warnings.
    set(COMMON_MSVC_FLAGS "/Wv:18")
  endif()
  string(APPEND COMMON_MSVC_FLAGS " /bigobj")
  # To keep MSVC from oversubscribing the CPU, force it to single threaded mode
  # msbuild/ninja will queue as many compile units as there are cores, no need for
  # MSVC to be internally threading as well.
  string(APPEND COMMON_MSVC_FLAGS " /cgthreads1 ")

  if(WITH_OPTIMIZED_DEBUG)
    set(BLENDER_CMAKE_C_FLAGS_DEBUG "/MDd ${COMMON_MSVC_FLAGS} /O2 /Ob2 /D_DEBUG /DPSAPI_VERSION=2 /DTINYFORMAT_ALLOW_WCHAR_STRINGS")
  else()
    set(BLENDER_CMAKE_C_FLAGS_DEBUG "/MDd ${COMMON_MSVC_FLAGS} /Zi /Ob0 /Od /RTC1 /D_DEBUG /DPSAPI_VERSION=2 /DTINYFORMAT_ALLOW_WCHAR_STRINGS")
  endif()
  set(BLENDER_CMAKE_C_FLAGS_MINSIZEREL "/MD ${COMMON_MSVC_FLAGS} /O1 /Ob1 /D NDEBUG /DPSAPI_VERSION=2 /DTINYFORMAT_ALLOW_WCHAR_STRINGS")
  set(BLENDER_CMAKE_C_FLAGS_RELEASE "/MD ${COMMON_MSVC_FLAGS} /O2 /Ob2 /DNDEBUG /DPSAPI_VERSION=2 /DTINYFORMAT_ALLOW_WCHAR_STRINGS")
  set(BLENDER_CMAKE_C_FLAGS_RELWITHDEBINFO "/MD ${COMMON_MSVC_FLAGS} /Zi /O2 /Ob1 /D NDEBUG /DPSAPI_VERSION=2 /DTINYFORMAT_ALLOW_WCHAR_STRINGS")

  if(WITH_OPTIMIZED_DEBUG)
    set(BLENDER_CMAKE_CXX_FLAGS_DEBUG "/MDd ${COMMON_MSVC_FLAGS} /D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS /O2 /Ob2 /D_DEBUG /D PLATFORM_WINDOWS /DPSAPI_VERSION=2 /DTINYFORMAT_ALLOW_WCHAR_STRINGS")
  else()
    set(BLENDER_CMAKE_CXX_FLAGS_DEBUG "/D_DEBUG /D PLATFORM_WINDOWS /D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS /MTd  ${COMMON_MSVC_FLAGS} /Zi /Ob0 /Od /RTC1 /DPSAPI_VERSION=2 /DTINYFORMAT_ALLOW_WCHAR_STRINGS")
  endif()
  set(BLENDER_CMAKE_CXX_FLAGS_MINSIZEREL "/MD /${COMMON_MSVC_FLAGS} /D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS /O1 /Ob1 /D NDEBUG  /D PLATFORM_WINDOWS /DPSAPI_VERSION=2 /DTINYFORMAT_ALLOW_WCHAR_STRINGS")
  set(BLENDER_CMAKE_CXX_FLAGS_RELEASE "/MD ${COMMON_MSVC_FLAGS} /D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS /O2 /Ob2 /D NDEBUG /D PLATFORM_WINDOWS /DPSAPI_VERSION=2 /DTINYFORMAT_ALLOW_WCHAR_STRINGS")
  set(BLENDER_CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MD ${COMMON_MSVC_FLAGS} /D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS /Zi /O2 /Ob1 /D NDEBUG /D PLATFORM_WINDOWS /DPSAPI_VERSION=2 /DTINYFORMAT_ALLOW_WCHAR_STRINGS")

  # Set similar flags for CLANG compilation.
  set(COMMON_CLANG_FLAGS "-D_DLL -D_MT") # Equivalent to MSVC /MD

  if(WITH_OPTIMIZED_DEBUG)
    set(BLENDER_CLANG_CMAKE_C_FLAGS_DEBUG "${COMMON_CLANG_FLAGS} -Xclang --dependent-lib=msvcrtd -O2 -D_DEBUG -DPSAPI_VERSION=2 -DTINYFORMAT_ALLOW_WCHAR_STRINGS")
  else()
    set(BLENDER_CLANG_CMAKE_C_FLAGS_DEBUG "${COMMON_CLANG_FLAGS} -Xclang --dependent-lib=msvcrtd -g -D_DEBUG -DPSAPI_VERSION=2 -DTINYFORMAT_ALLOW_WCHAR_STRINGS")
  endif()
  set(BLENDER_CLANG_CMAKE_C_FLAGS_MINSIZEREL "${COMMON_CLANG_FLAGS} -Xclang --dependent-lib=msvcrt -Os -DNDEBUG -DPSAPI_VERSION=2 -DTINYFORMAT_ALLOW_WCHAR_STRINGS")
  set(BLENDER_CLANG_CMAKE_C_FLAGS_RELEASE "${COMMON_CLANG_FLAGS}  -Xclang --dependent-lib=msvcrt -O2 -DNDEBUG -DPSAPI_VERSION=2 -DTINYFORMAT_ALLOW_WCHAR_STRINGS")
  set(BLENDER_CLANG_CMAKE_C_FLAGS_RELWITHDEBINFO "${COMMON_CLANG_FLAGS} -Xclang --dependent-lib=msvcrt -g -O2 -DNDEBUG -DPSAPI_VERSION=2 -DTINYFORMAT_ALLOW_WCHAR_STRINGS")

  if(WITH_OPTIMIZED_DEBUG)
    set(BLENDER_CLANG_CMAKE_CXX_FLAGS_DEBUG "${COMMON_CLANG_FLAGS} -Xclang --dependent-lib=msvcrtd -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS -O2 -D_DEBUG -DPLATFORM_WINDOWS -DPSAPI_VERSION=2 -DTINYFORMAT_ALLOW_WCHAR_STRINGS")
  else()
    set(BLENDER_CLANG_CMAKE_CXX_FLAGS_DEBUG "${COMMON_CLANG_FLAG} -Xclang --dependent-lib=msvcrtd -D_DEBUG -DPLATFORM_WINDOWS -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS -g -DPSAPI_VERSION=2 -DTINYFORMAT_ALLOW_WCHAR_STRINGS")
  endif()
  set(BLENDER_CLANG_CMAKE_CXX_FLAGS_MINSIZEREL "${COMMON_CLANG_FLAGS} -Xclang --dependent-lib=msvcrt -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS -O2 -DNDEBUG  -DPLATFORM_WINDOWS -DPSAPI_VERSION=2 -DTINYFORMAT_ALLOW_WCHAR_STRINGS")
  set(BLENDER_CLANG_CMAKE_CXX_FLAGS_RELEASE "${COMMON_CLANG_FLAGS} -Xclang --dependent-lib=msvcrt -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS -O2 -DNDEBUG -DPLATFORM_WINDOWS -DPSAPI_VERSION=2 -DTINYFORMAT_ALLOW_WCHAR_STRINGS")
  set(BLENDER_CLANG_CMAKE_CXX_FLAGS_RELWITHDEBINFO "${COMMON_CLANG_FLAGS} -Xclang --dependent-lib=msvcrt -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS -g -O2 -DNDEBUG -DPLATFORM_WINDOWS -DPSAPI_VERSION=2 -DTINYFORMAT_ALLOW_WCHAR_STRINGS")

  set(DEFAULT_CLANG_CMAKE_FLAGS
    -DCMAKE_BUILD_TYPE=${BUILD_MODE}
    -DCMAKE_C_FLAGS_DEBUG=${BLENDER_CLANG_CMAKE_C_FLAGS_DEBUG}
    -DCMAKE_C_FLAGS_MINSIZEREL=${BLENDER_CLANG_CMAKE_C_FLAGS_MINSIZEREL}
    -DCMAKE_C_FLAGS_RELEASE=${BLENDER_CLANG_CMAKE_C_FLAGS_RELEASE}
    -DCMAKE_C_FLAGS_RELWITHDEBINFO=${BLENDER_CLANG_CMAKE_C_FLAGS_RELWITHDEBINFO}
    -DCMAKE_CXX_FLAGS_DEBUG=${BLENDER_CLANG_CMAKE_CXX_FLAGS_DEBUG}
    -DCMAKE_CXX_FLAGS_MINSIZEREL=${BLENDER_CLANG_CMAKE_CXX_FLAGS_MINSIZEREL}
    -DCMAKE_CXX_FLAGS_RELEASE=${BLENDER_CLANG_CMAKE_CXX_FLAGS_RELEASE}
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=${BLENDER_CLANG_CMAKE_CXX_FLAGS_RELWITHDEBINFO}
    -DCMAKE_CXX_STANDARD=17
  )

  set(PLATFORM_FLAGS)
  set(PLATFORM_CXX_FLAGS)

  if(BLENDER_PLATFORM_ARM)
    # In some cases on ARM64 (unsure why), dep builds using the "Ninja" generator appear to use
    # the x86 host tools (ie, x86 cl.exe producing ARM64 binaries). This is problematic when
    # building things like LLVM, as memory is limited to 3GB, giving internal compiler errors.
    # Here, we set CMAKE_C_COMPILER et al via PLATFORM_CMAKE_FLAGS to point to the ARM64 native
    # binary, which doesn't have this issue.
    # We make an assumption that the tools (ie, right now in the code) are the ones we want
    set(PLATFORM_CMAKE_FLAGS
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_AR=${CMAKE_AR}
      -DCMAKE_LINKER=${CMAKE_LINKER}
      -DCMAKE_MT=${CMAKE_MT}
      -DCMAKE_RC_COMPILER=${CMAKE_RC_COMPILER}
    )
  else()
    set(PLATFORM_CMAKE_FLAGS)
  endif()

  set(MINGW_PATH ${DOWNLOAD_DIR}/msys2/msys64/)
  set(MINGW_SHELL ming64sh.cmd)
  set(PERL_SHELL ${DOWNLOAD_DIR}/perl/portableshell.bat)
  set(MINGW_HOST x86_64-w64-mingw32)

  set(MINGW_CFLAGS)
  set(MINGW_LDFLAGS)

  # some build systems like meson will respect the *nix like environment vars
  # like CFLAGS and LDFlags but will still build with the MSVC compiler, so for
  # those we need to empty out the gcc style flags that are normally in there.
  set(CONFIGURE_ENV_MSVC
    cd ${MINGW_PATH} &&
    call ${PERL_SHELL} &&
    call ${MINGW_SHELL} &&
    set path &&
    set CFLAGS= &&
    set LDFLAGS=
  )

  set(CONFIGURE_ENV
    cd ${MINGW_PATH} &&
    call ${PERL_SHELL} &&
    call ${MINGW_SHELL} &&
    set path &&
    set CC=cl &&
    set CXX=cl &&
    set CFLAGS=${MINGW_CFLAGS} &&
    set LDFLAGS=${MINGW_LDFLAGS}
  )

  set(CONFIGURE_ENV_NO_PERL
    cd ${MINGW_PATH} &&
    call ${MINGW_SHELL} &&
    set path &&
    set CC=cl &&
    set CXX=cl &&
    set LD=link &&
    set CFLAGS=${MINGW_CFLAGS} &&
    set LDFLAGS=${MINGW_LDFLAGS}
  )

  set(CONFIGURE_ENV_CLANG_CL_NO_PERL
    cd ${MINGW_PATH} &&
    call ${MINGW_SHELL} &&
    set path &&
    set CC=${LIBDIR}/llvm/bin/clang-cl.exe &&
    set CXX=${LIBDIR}/llvm/bin/clang-cl.exe &&
    set RANLIB=${LIBDIR}/llvm/bin/llvm-ranlib.exe &&
    set RC=${LIBDIR}/llvm/bin/llvm-rc.exe &&
    set AR=${LIBDIR}/llvm/bin/llvm-ar.exe &&
    set CFLAGS=${MINGW_CFLAGS} &&
    set LDFLAGS=${MINGW_LDFLAGS}
  )

  set(CONFIGURE_COMMAND sh ./configure)
  set(CONFIGURE_COMMAND_NO_TARGET ${CONFIGURE_COMMAND})
else()
  set(PATCH_CMD patch)
  set(LIBEXT ".a")
  set(LIBPREFIX "lib")
  set(MESON ${LIBDIR}/python/bin/meson)
  if(APPLE)
    set(SHAREDLIBEXT ".dylib")

    # Use same Xcode detection as Blender itself.
    include(../cmake/platform/platform_apple_xcode.cmake)

    if("${CMAKE_OSX_ARCHITECTURES}" STREQUAL "arm64")
      set(BLENDER_PLATFORM_ARM ON)
    endif()

    set(PLATFORM_CFLAGS "-isysroot ${CMAKE_OSX_SYSROOT} -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET} -arch ${CMAKE_OSX_ARCHITECTURES}")
    set(PLATFORM_CXXFLAGS "-isysroot ${CMAKE_OSX_SYSROOT} -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET} -std=c++17 -stdlib=libc++ -arch ${CMAKE_OSX_ARCHITECTURES}")
    set(PLATFORM_LDFLAGS "-isysroot ${CMAKE_OSX_SYSROOT} -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET} -arch ${CMAKE_OSX_ARCHITECTURES} -headerpad_max_install_names")
    if("${CMAKE_OSX_ARCHITECTURES}" STREQUAL "x86_64")
      set(PLATFORM_BUILD_TARGET --build=x86_64-apple-darwin19.0.0) # OS X 10.15
    else()
      set(PLATFORM_BUILD_TARGET --build=aarch64-apple-darwin20.0.0) # macOS 11.00
    endif()
    set(PLATFORM_CMAKE_FLAGS
      -DCMAKE_OSX_ARCHITECTURES:STRING=${CMAKE_OSX_ARCHITECTURES}
      -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=${CMAKE_OSX_DEPLOYMENT_TARGET}
      -DCMAKE_OSX_SYSROOT:PATH=${CMAKE_OSX_SYSROOT}
    )
  else()
    set(SHAREDLIBEXT ".so")

    if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
      set(BLENDER_PLATFORM_ARM ON)
    endif()

    set(PLATFORM_CFLAGS "-fPIC")
    set(PLATFORM_CXXFLAGS "-std=c++17 -fPIC")
    set(PLATFORM_LDFLAGS)
    set(PLATFORM_BUILD_TARGET)
    set(PLATFORM_CMAKE_FLAGS -DCMAKE_INSTALL_LIBDIR=lib)
  endif()

  if(WITH_OPTIMIZED_DEBUG)
    set(BLENDER_CMAKE_C_FLAGS_DEBUG "-O2 -DNDEBUG ${PLATFORM_CFLAGS}")
  else()
    set(BLENDER_CMAKE_C_FLAGS_DEBUG "-g ${PLATFORM_CFLAGS}")
  endif()
  set(BLENDER_CMAKE_C_FLAGS_MINSIZEREL "-Os -DNDEBUG ${PLATFORM_CFLAGS}")
  set(BLENDER_CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG ${PLATFORM_CFLAGS}")
  set(BLENDER_CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG ${PLATFORM_CFLAGS}")

  if(WITH_OPTIMIZED_DEBUG)
    set(BLENDER_CMAKE_CXX_FLAGS_DEBUG "-O2 -DNDEBUG ${PLATFORM_CXXFLAGS}")
  else()
    set(BLENDER_CMAKE_CXX_FLAGS_DEBUG "-g ${PLATFORM_CXXFLAGS}")
  endif()

  set(BLENDER_CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG ${PLATFORM_CXXFLAGS}")
  set(BLENDER_CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG ${PLATFORM_CXXFLAGS}")
  set(BLENDER_CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG ${PLATFORM_CXXFLAGS}")

  set(CONFIGURE_ENV
    export MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET} &&
    export MACOSX_SDK_VERSION=${CMAKE_OSX_DEPLOYMENT_TARGET} &&
    export CFLAGS=${PLATFORM_CFLAGS} &&
    export CXXFLAGS=${PLATFORM_CXXFLAGS} &&
    export LDFLAGS=${PLATFORM_LDFLAGS}
  )
  set(CONFIGURE_ENV_NO_PERL ${CONFIGURE_ENV})
  set(CONFIGURE_COMMAND ./configure ${PLATFORM_BUILD_TARGET})
  set(CONFIGURE_COMMAND_NO_TARGET ./configure)
endif()

set(DEFAULT_CMAKE_FLAGS
  -DCMAKE_BUILD_TYPE=${BUILD_MODE}
  -DCMAKE_C_FLAGS_DEBUG=${BLENDER_CMAKE_C_FLAGS_DEBUG}
  -DCMAKE_C_FLAGS_MINSIZEREL=${BLENDER_CMAKE_C_FLAGS_MINSIZEREL}
  -DCMAKE_C_FLAGS_RELEASE=${BLENDER_CMAKE_C_FLAGS_RELEASE}
  -DCMAKE_C_FLAGS_RELWITHDEBINFO=${BLENDER_CMAKE_C_FLAGS_RELWITHDEBINFO}
  -DCMAKE_CXX_FLAGS_DEBUG=${BLENDER_CMAKE_CXX_FLAGS_DEBUG}
  -DCMAKE_CXX_FLAGS_MINSIZEREL=${BLENDER_CMAKE_CXX_FLAGS_MINSIZEREL}
  -DCMAKE_CXX_FLAGS_RELEASE=${BLENDER_CMAKE_CXX_FLAGS_RELEASE}
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
  -DCMAKE_CXX_STANDARD=17
  ${PLATFORM_CMAKE_FLAGS}
)

if(WIN32)
  if(BUILD_MODE STREQUAL Debug)
    set(ZLIB_LIBRARY zlibstaticd${LIBEXT})
  else()
    set(ZLIB_LIBRARY zlibstatic${LIBEXT})
  endif()
else()
  set(ZLIB_LIBRARY libz${LIBEXT})
endif()

if(MSVC)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

set(CMAKE_INSTALL_MESSAGE LAZY)

# On windows we sometimes want to build with ninja, but not all projects quite
# yet, so for select project we pass PLATFORM_ALT_GENERATOR as the generator
if(WIN32)
  set(PLATFORM_ALT_GENERATOR "Ninja")
else()
  set(PLATFORM_ALT_GENERATOR "Unix Makefiles")
endif()
