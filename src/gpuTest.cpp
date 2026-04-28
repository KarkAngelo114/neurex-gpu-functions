#include <napi.h>
#include <CL/cl.h>
#include <vector>
#include <string>

struct GPUInfo {
    std::string name;
    std::string vendor;
    std::string driverVersion;
    std::string openclVersion;
    std::string platformName;
    cl_ulong       globalMemBytes   = 0;
    cl_uint        computeUnits     = 0;
    cl_uint        maxClockMHz      = 0;
    cl_bool        hostUnifiedMemory= CL_FALSE;
    cl_device_type deviceType       = 0;
};

struct DetectionResult {
    bool ok = true;
    std::string error;          // empty if ok
    cl_uint platformCount = 0;
    std::vector<GPUInfo> devices;
};


static std::string getDeviceString(cl_device_id dev, cl_device_info param) {
    size_t size = 0;
    if (clGetDeviceInfo(dev, param, 0, nullptr, &size) != CL_SUCCESS || size == 0)
        return {};
    std::string out(size, '\0');
    if (clGetDeviceInfo(dev, param, size, out.data(), nullptr) != CL_SUCCESS)
        return {};
    if (!out.empty() && out.back() == '\0') out.pop_back();
    return out;
}

static std::string getPlatformString(cl_platform_id p, cl_platform_info param) {
    size_t size = 0;
    if (clGetPlatformInfo(p, param, 0, nullptr, &size) != CL_SUCCESS || size == 0)
        return {};
    std::string out(size, '\0');
    if (clGetPlatformInfo(p, param, size, out.data(), nullptr) != CL_SUCCESS)
        return {};
    if (!out.empty() && out.back() == '\0') out.pop_back();
    return out;
}

DetectionResult detectOpenCLDevices() {
    DetectionResult result;

    cl_uint platformCount = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &platformCount);
    if (err != CL_SUCCESS) {
        result.ok = false;
        result.error = "clGetPlatformIDs failed (code " + std::to_string(err) + ")";
        return result;
    }
    result.platformCount = platformCount;
    if (platformCount == 0) return result; // ok=true, empty devices

    std::vector<cl_platform_id> platforms(platformCount);
    err = clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        result.ok = false;
        result.error = "clGetPlatformIDs (fetch) failed (code " + std::to_string(err) + ")";
        return result;
    }

    for (auto platform : platforms) {
        std::string platformName = getPlatformString(platform, CL_PLATFORM_NAME);

        cl_uint deviceCount = 0;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);
        // CL_DEVICE_NOT_FOUND just means this platform has no GPU; skip quietly
        if (err == CL_DEVICE_NOT_FOUND || deviceCount == 0) continue;
        if (err != CL_SUCCESS) continue;

        std::vector<cl_device_id> devices(deviceCount);
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr);
        if (err != CL_SUCCESS) continue;

        for (auto device : devices) {
            GPUInfo info;
            info.platformName   = platformName;
            info.name           = getDeviceString(device, CL_DEVICE_NAME);
            info.vendor         = getDeviceString(device, CL_DEVICE_VENDOR);
            info.driverVersion  = getDeviceString(device, CL_DRIVER_VERSION);
            info.openclVersion  = getDeviceString(device, CL_DEVICE_VERSION);

            clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
                            sizeof(cl_ulong), &info.globalMemBytes, nullptr);
            clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(cl_uint),  &info.computeUnits, nullptr);
            clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                            sizeof(cl_uint),  &info.maxClockMHz, nullptr);
            clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY,
                            sizeof(cl_bool),  &info.hostUnifiedMemory, nullptr);
            clGetDeviceInfo(device, CL_DEVICE_TYPE,
                            sizeof(cl_device_type), &info.deviceType, nullptr);

            result.devices.push_back(std::move(info));
        }
    }

    return result;
}

static const char* deviceTypeToString(cl_device_type t) {
    if (t & CL_DEVICE_TYPE_GPU)         return "gpu";
    if (t & CL_DEVICE_TYPE_CPU)         return "cpu";
    if (t & CL_DEVICE_TYPE_ACCELERATOR) return "accelerator";
    return "unknown";
}

Napi::Value CheckGPUWrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    DetectionResult det = detectOpenCLDevices();

    Napi::Object out = Napi::Object::New(env);
    out.Set("ok",            Napi::Boolean::New(env, det.ok));
    out.Set("error",         Napi::String::New(env, det.error));
    out.Set("platformCount", Napi::Number::New(env, det.platformCount));

    Napi::Array arr = Napi::Array::New(env, det.devices.size());
    for (size_t i = 0; i < det.devices.size(); i++) {
        const auto& d = det.devices[i];
        Napi::Object data = Napi::Object::New(env);

        data.Set("gpu",                Napi::String::New(env, d.name));
        data.Set("vendor",             Napi::String::New(env, d.vendor));
        data.Set("platform",           Napi::String::New(env, d.platformName));
        data.Set("driverVersion",      Napi::String::New(env, d.driverVersion));
        data.Set("openclVersion",      Napi::String::New(env, d.openclVersion));
        data.Set("deviceType",         Napi::String::New(env, deviceTypeToString(d.deviceType)));

        // numeric fields — use BigInt for memory so 4GB+ doesn't lose precision
        data.Set("globalMemBytes",     Napi::BigInt::New(env, (uint64_t)d.globalMemBytes));
        data.Set("computeUnits",       Napi::Number::New(env, d.computeUnits));
        data.Set("maxClockMHz",        Napi::Number::New(env, d.maxClockMHz));
        data.Set("hostUnifiedMemory",  Napi::Boolean::New(env, d.hostUnifiedMemory == CL_TRUE));

        arr[i] = data;
    }
    out.Set("devices", arr);

    return out;
}

void detectGPU(Napi::Env env, Napi::Object exports) {
    exports.Set("Detect_GPU", Napi::Function::New(env, CheckGPUWrapper));
}