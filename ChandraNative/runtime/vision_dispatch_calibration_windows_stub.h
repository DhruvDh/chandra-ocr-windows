// New ChandraNative code, MPL-2.0. Test-only POSIX stand-in for the six Win32 receipt calls used by
// vision_dispatch_calibration.cpp, so the unchanged entry can run on the CPU harness. The test script
// copies it to FRESH_BUILD/stub/windows.h. Never on a Windows include path; not a Win32 emulation.
#pragma once
#include <cstdint>
#include <filesystem>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

using BOOL = int;
using DWORD = uint32_t;
using HANDLE = int*;
struct LARGE_INTEGER { int64_t QuadPart; };
inline int stubInvalidHandle = -1;
#define INVALID_HANDLE_VALUE (&stubInvalidHandle)
constexpr DWORD GENERIC_WRITE = 0x40000000u, FILE_SHARE_READ = 1u, CREATE_NEW = 1u, FILE_ATTRIBUTE_NORMAL = 0x80u, FILE_BEGIN = 0u;

inline HANDLE CreateFileW(const std::filesystem::path::value_type* name, DWORD access, DWORD, void*, DWORD disposition, DWORD, HANDLE) {
    if (access != GENERIC_WRITE || disposition != CREATE_NEW) return INVALID_HANDLE_VALUE;
    const int fd = open(name, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    return fd < 0 ? INVALID_HANDLE_VALUE : new int(fd);
}
inline BOOL SetFilePointerEx(HANDLE file, LARGE_INTEGER distance, void*, DWORD method) {
    return method == FILE_BEGIN && lseek(*file, off_t(distance.QuadPart), SEEK_SET) == off_t(distance.QuadPart);
}
inline BOOL WriteFile(HANDLE file, const void* data, DWORD size, DWORD* written, void*) {
    const char* p = static_cast<const char*>(data); DWORD done = 0;
    while (done < size) {
        const ssize_t n = write(*file, p + done, size - done);
        if (n <= 0) break;
        done += DWORD(n);
    }
    *written = done; return done == size;
}
inline BOOL SetEndOfFile(HANDLE file) { const off_t at = lseek(*file, 0, SEEK_CUR); return at >= 0 && ftruncate(*file, at) == 0; }
inline BOOL FlushFileBuffers(HANDLE file) { return fsync(*file) == 0; }
inline BOOL CloseHandle(HANDLE file) { const int result = close(*file); delete file; return result == 0; }
