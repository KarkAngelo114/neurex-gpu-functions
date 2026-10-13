


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
        "src/math.cpp",
        "src/pooling.cpp",
        "src/gpuTest.cpp",
        "src/utils.cpp",
        "src/init.cpp",
        "src/gpu/gpu_context.cpp",
        "src/globals/globals.cpp",
        "src/functions/extendedNapiFunc.cpp",
        "src/embeddings.cpp",
        "src/recurrent.cpp",
        "src/loss.cpp",
        "src/normalizers.cpp",
        "src/transConv.cpp",
        "src/attention.cpp",
        "src/encoding.cpp"
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
      "conditions": [
        ["OS=='linux'", {
          "cflags_cc": [ "-std=c++17", "-fexceptions", "-O3", "-ffast-math", "-march=x86-64-v3", "-fopenmp" ],
          "ldflags": [ "-fopenmp" ],
          "libraries": [ "-lOpenCL" ],
        }],
        ["OS=='mac'", {
          "defines": [
            "CL_SILENCE_DEPRECATION"
          ],
          "include_dirs": [
            "<!(brew --prefix libomp)/include"
          ],
          "libraries": [
            "-framework OpenCL",
            "<!(brew --prefix libomp)/lib/libomp.a"
          ],
          "xcode_settings": {
            "CLANG_CXX_LANGUAGE_STANDARD": "c++17",
            "CLANG_CXX_LIBRARY": "libc++",
            "MACOSX_DEPLOYMENT_TARGET": "11.0",
            "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
            "GCC_OPTIMIZATION_LEVEL": "3",
            "OTHER_CPLUSPLUSFLAGS": [ "-Xpreprocessor", "-fopenmp", "-ffast-math" ],
            "OTHER_CFLAGS": [ "-ffast-math" ]
          },
          "conditions": [
            ["target_arch=='x64'", {
              "xcode_settings": {
                "OTHER_CPLUSPLUSFLAGS": [ "-Xpreprocessor", "-fopenmp", "-march=x86-64-v3", "-ffast-math" ]
              }
            }]
          ]
        }],
        ["OS=='win'", {
          "msvs_settings": {
            "VCCLCompilerTool": {
              "ExceptionHandling": 1,
              "AdditionalOptions": [
                "/O2",
                "/openmp"
              ],
              "RuntimeLibrary": 2
            },
            "VCLinkerTool": {
              "AdditionalLibraryDirectories": [
                "<!(node -p \"const fs=require('fs'); const path=require('path'); const candidates=[process.env.OPENCL_LIB_DIR && path.join(process.env.OPENCL_LIB_DIR,'OpenCL.lib'), process.env.OPENCL_LIB, 'C:/Program Files (x86)/Windows Kits/10/Lib', 'C:/Program Files/Windows Kits/10/Lib']; const match = candidates.find(v => !!v && fs.existsSync(v)); match || ''\")"
              ]
            }
          },
          "libraries": [
            "<!(node -p \"const fs=require('fs'); const path=require('path'); const candidates=[process.env.OPENCL_LIB, process.env.OPENCL_LIB_DIR && path.join(process.env.OPENCL_LIB_DIR,'OpenCL.lib'), 'C:/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0/um/x64/OpenCL.lib', 'C:/Program Files/Windows Kits/10/Lib/10.0.26100.0/um/x64/OpenCL.lib', 'C:/Program Files (x86)/Windows Kits/10/Lib/10.0.19041.0/um/x64/OpenCL.lib', 'C:/Program Files/Windows Kits/10/Lib/10.0.19041.0/um/x64/OpenCL.lib']; const match = candidates.find(v => !!v && fs.existsSync(v)); match || 'OpenCL.lib';\")"
          ],
          "defines": [ "_HAS_EXCEPTIONS=1" ]
        }]
      ]
    }
  ]
}
