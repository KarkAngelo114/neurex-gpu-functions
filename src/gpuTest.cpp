#include <napi.h>
#include <CL/cl.h>
#include <vector>
#include <string>

#include <napi.h>
#include <CL/cl.h>
#include <algorithm>
#include <string>
#include <vector>

struct GPUInfo {
    uint32_t index = 0;
    std::string name;
    std::string vendor;
    std::string driverVersion;
    std::string openclVersion;
    std::string platformName;
    cl_ulong globalMemBytes = 0;
    cl_uint computeUnits = 0;
    cl_uint maxClockMHz = 0;
    cl_bool hostUnifiedMemory = CL_FALSE;
    cl_device_type deviceType = 0;
};

struct DetectionResult {
    bool ok = true;
    std::string error;
    cl_uint platformCount = 0;
    std::vector<GPUInfo> devices;
};

struct EnumeratedDevice {
    cl_platform_id platform;
    cl_device_id   device;
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

static const char* deviceTypeToString(cl_device_type t) {
    if (t & CL_DEVICE_TYPE_GPU)         return "gpu";
    if (t & CL_DEVICE_TYPE_CPU)         return "cpu";
    if (t & CL_DEVICE_TYPE_ACCELERATOR) return "accelerator";
    return "unknown";
}

std::vector<EnumeratedDevice> enumerateGPUDevices(cl_uint* platformCountOut, cl_int* errOut) {
    std::vector<EnumeratedDevice> result;
    cl_uint platformCount = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &platformCount);
    if (platformCountOut) *platformCountOut = platformCount;
    if (errOut) *errOut = err;
    if (err != CL_SUCCESS || platformCount == 0) return result;

    std::vector<cl_platform_id> platforms(platformCount);
    err = clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    if (errOut) *errOut = err;
    if (err != CL_SUCCESS) return result;

    for (auto platform : platforms) {
        cl_uint deviceCount = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount) != CL_SUCCESS
            || deviceCount == 0) continue;

        std::vector<cl_device_id> devices(deviceCount);
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr) != CL_SUCCESS)
            continue;

        for (auto d : devices) result.push_back({platform, d});
    }
    return result;
}

DetectionResult detectOpenCLDevices() {
    DetectionResult result;
    cl_int err = CL_SUCCESS;
    auto found = enumerateGPUDevices(&result.platformCount, &err);

    if (err != CL_SUCCESS) {
        result.ok = false;
        result.error = "OpenCL platform enumeration failed (code " + std::to_string(err) + ")";
        return result;
    }

    uint32_t idx = 0;
    for (auto& e : found) {
        GPUInfo info;
        info.index          = idx++;
        info.platformName   = getPlatformString(e.platform, CL_PLATFORM_NAME);
        info.name           = getDeviceString(e.device, CL_DEVICE_NAME);
        info.vendor         = getDeviceString(e.device, CL_DEVICE_VENDOR);
        info.driverVersion  = getDeviceString(e.device, CL_DRIVER_VERSION);
        info.openclVersion  = getDeviceString(e.device, CL_DEVICE_VERSION);

        clGetDeviceInfo(e.device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &info.globalMemBytes, nullptr);
        clGetDeviceInfo(e.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &info.computeUnits, nullptr);
        clGetDeviceInfo(e.device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &info.maxClockMHz, nullptr);
        clGetDeviceInfo(e.device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &info.hostUnifiedMemory, nullptr);
        clGetDeviceInfo(e.device, CL_DEVICE_TYPE, sizeof(cl_device_type), &info.deviceType, nullptr);

        result.devices.push_back(std::move(info));
    }

    return result;
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
        data.Set("index", Napi::Number::New(env, d.index));

        arr[i] = data;
    }

    out.Set("devices", arr);

    return out;
}

void detectGPU(Napi::Env env, Napi::Object exports) {
    exports.Set("Detect_GPU", Napi::Function::New(env, CheckGPUWrapper));
}