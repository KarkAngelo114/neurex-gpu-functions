
{
  "variables": {
    "eigen_dir%": "deps/eigen"
  },
  "targets": [
    {
      "target_name": "neurex-core-native",
      "sources": [
        "src/main.cpp",
        "src/matmul.cpp",
        "src/activations.cpp",
        "src/convolve.cpp",
        "src/gradientsComputation.cpp",
        "src/optimizer_internals.cpp",
        "src/gradientScaler.cpp",
        "src/math.cpp",
        "src/pooling.cpp",
        "src/gpuTest.cpp",
        "src/utils.cpp",
        "src/init.cpp",
        "src/gpu/gpu_context.cpp"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "<(eigen_dir)",
        "<(module_root_dir)/deps/opencl"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "defines": [
        "NAPI_CPP_EXCEPTIONS",
        "NAPI_VERSION=8",
        "EIGEN_MPL2_ONLY",
        "EIGEN_NO_DEBUG",
        "CL_TARGET_OPENCL_VERSION=120"
      ],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "cflags_cc": [ "-std=c++17", "-fexceptions", "-O3", "-ffast-math" ],
      "conditions": [
        ["OS=='linux'", {
          "cflags_cc": [ "-march=x86-64-v3", "-fopenmp" ],
          "ldflags": [ "-fopenmp" ],
          "libraries": [ "-lOpenCL" ],
        }],
        ["OS=='mac'", {
          "defines": [ 
            "CL_SILENCE_DEPRECATION"
          ],
          "libraries": [
            "-framework OpenCL"
          ],
          "xcode_settings": {
            "CLANG_CXX_LANGUAGE_STANDARD": "c++17",
            "CLANG_CXX_LIBRARY": "libc++",
            "MACOSX_DEPLOYMENT_TARGET": "11.0",
            "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
            "GCC_OPTIMIZATION_LEVEL": "3",
            "OTHER_CPLUSPLUSFLAGS": [ "-ffast-math" ],
            "OTHER_CFLAGS": [ "-ffast-math" ]
          },
          "conditions": [
            ["target_arch=='x64'", {
              "xcode_settings": { "OTHER_CPLUSPLUSFLAGS": [ "-march=x86-64-v3", "-ffast-math" ] }
            }]
          ]
        }],
        ["OS=='win'", {
          "msvs_settings": {
            "VCCLCompilerTool": {
              "ExceptionHandling": 1,
              "AdditionalOptions": [ "/std:c++17", "/O2", "/arch:AVX2", "/fp:fast", "/EHsc" ],
              "RuntimeLibrary": 2
            },
            "VCLinkerTool": {
              "AdditionalLibraryDirectories": [
                "<!(echo %VCPKG_INSTALLATION_ROOT%)/installed/x64-windows/lib"
              ]
            }
          },
          "libraries": [ "OpenCL.lib" ],
          "defines": [ "_HAS_EXCEPTIONS=1" ]
        }]
      ]
    }
  ]
}