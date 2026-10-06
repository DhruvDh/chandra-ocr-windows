// Test-only, MPL-2.0. POSIX implementation of the Win32 subset declared in win32/windows.h.
// Handles wrap file descriptors; symlinks report FILE_ATTRIBUTE_REPARSE_POINT like Windows reparse
// points; CREATE_NEW is O_CREAT|O_EXCL; SHA-256 uses diagnostics::Sha256. Share modes, ACLs and
// pipe FlushFileBuffers semantics are not emulated. CHANDRA_SHIM_JOB describes the observed Job:
// unset means "not in a Job"; otherwise comma-separated kill, process_memory=N, job_memory=N, breakaway.
// ReadFile follows the synchronous Win32 contract the worker's stdin reader depends on: a pipe whose
// writer closed fails with ERROR_BROKEN_PIPE (as an anonymous pipe does) while a file or device ends with a
// successful zero-byte read; a read blocked on a pipe can be canceled by CancelSynchronousIo, completing
// with ERROR_OPERATION_ABORTED, and CancelSynchronousIo reports ERROR_NOT_FOUND when nothing is pending.
// POSIX read errors map to the Win32 status for the same condition; an errno with no such status becomes
// 0x20000000|errno, an application-defined code no system status uses. CHANDRA_SHIM_READ_FAIL=N:CODE makes
// the Nth standard-input ReadFile fail with Win32 CODE without reading, for statuses Linux cannot produce.
// CHANDRA_SHIM_READ_GATE=N:PATH makes the Nth standard-input ReadFile first wait, bounded, until PATH exists
// (marking PATH.waiting), so a test can append to a regular-file stdin before its genuine zero-byte end.
#include "win32/windows.h"
#include "win32/bcrypt.h"
#include "win32/psapi.h"
#include "../../../ChandraNative/runtime/diagnostics.h"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <mutex>
#include <poll.h>
#include <pthread.h>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
thread_local DWORD lastError = 0;
HANDLE encode(int fd) { return reinterpret_cast<HANDLE>(intptr_t(fd) + 0x100); }
int decode(HANDLE h) { return int(reinterpret_cast<intptr_t>(h) - 0x100); }
DWORD fail(DWORD code) { lastError = code; return code; }
std::wstring wide(const char* text) {
    std::wstring out; const auto* p = reinterpret_cast<const unsigned char*>(text);
    while (*p) {
        uint32_t c = *p++, extra = c >= 0xf0 ? 3 : c >= 0xe0 ? 2 : c >= 0xc0 ? 1 : 0;
        if (extra) c &= 0x3fu >> extra;
        for (uint32_t i = 0; i < extra && (*p & 0xc0) == 0x80; ++i) c = (c << 6) | (*p++ & 0x3f);
        out.push_back(wchar_t(c));
    }
    return out;
}
struct ShimJob { bool inJob = false; JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{}; };
ShimJob job() {
    ShimJob result; const char* text = std::getenv("CHANDRA_SHIM_JOB"); if (!text) return result;
    result.inJob = true; std::string all(text); size_t start = 0;
    while (start <= all.size()) {
        size_t end = all.find(',', start); if (end == std::string::npos) end = all.size();
        std::string item = all.substr(start, end - start); auto& basic = result.info.BasicLimitInformation;
        if (item == "kill") basic.LimitFlags |= JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        else if (item == "breakaway") basic.LimitFlags |= JOB_OBJECT_LIMIT_BREAKAWAY_OK;
        else if (item.rfind("process_memory=", 0) == 0) { basic.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY; result.info.ProcessMemoryLimit = std::strtoull(item.c_str() + 15, nullptr, 10); }
        else if (item.rfind("job_memory=", 0) == 0) { basic.LimitFlags |= JOB_OBJECT_LIMIT_JOB_MEMORY; result.info.JobMemoryLimit = std::strtoull(item.c_str() + 11, nullptr, 10); }
        start = end + 1;
    }
    return result;
}
// Threads currently blocked in a pipe ReadFile, each with the write end of its own wake pipe.
std::mutex pendingLock;
std::map<pthread_t, int> pendingReads;
struct WakePipe { int fds[2] = {-1, -1}; ~WakePipe() { for (int fd : fds) if (fd >= 0) close(fd); } };
thread_local WakePipe wake;
std::atomic<uint64_t> standardInputReads{0};
DWORD readStatus(int fd, int error) {
    if (error == EISDIR) return ERROR_INVALID_FUNCTION; // Windows refuses ReadFile on a directory handle.
    if (error == EBADF) { // Linux says EBADF for a write-only descriptor too; Windows says access denied.
        const int flags = fcntl(fd, F_GETFL);
        return flags >= 0 && (flags & O_ACCMODE) == O_WRONLY ? ERROR_ACCESS_DENIED : ERROR_INVALID_HANDLE;
    }
    if (error == EACCES || error == EPERM) return ERROR_ACCESS_DENIED;
    if (error == EAGAIN || error == EWOULDBLOCK) return ERROR_NO_DATA; // A nonblocking (PIPE_NOWAIT) pipe with nothing to read.
    return 0x20000000u | DWORD(error);
}
// The text after "N:" in variable `name` when this is the Nth standard-input read, else null.
const char* nthRead(const char* name, uint64_t n) {
    const char* spec = std::getenv(name); if (!spec) return nullptr;
    char* end = nullptr; const uint64_t at = std::strtoull(spec, &end, 10);
    return at == n && *end == ':' ? end + 1 : nullptr;
}
bool injectedReadFailure(int fd) {
    if (fd != 0) return false;
    const uint64_t n = ++standardInputReads;
    if (const char* path = nthRead("CHANDRA_SHIM_READ_GATE", n)) {
        { std::ofstream marker(std::string(path) + ".waiting"); marker << n << '\n'; }
        struct stat s{}; const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (stat(path, &s) != 0 && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const char* code = nthRead("CHANDRA_SHIM_READ_FAIL", n); if (!code) return false;
    fail(DWORD(std::strtoul(code, nullptr, 10))); return true;
}
// Waits until the pipe is readable or CancelSynchronousIo targets this thread; false when canceled. Data or
// end of file that is already available completes the read, as on Windows, where only a pending read is canceled.
bool awaitPipe(int fd) {
    if (wake.fds[0] < 0 && pipe2(wake.fds, O_CLOEXEC | O_NONBLOCK) != 0) return true; // Not cancelable, but still a correct read.
    { std::lock_guard<std::mutex> guard(pendingLock); pendingReads[pthread_self()] = wake.fds[1]; }
    pollfd fds[2] = {{fd, POLLIN, 0}, {wake.fds[0], POLLIN, 0}};
    while (poll(fds, 2, -1) < 0 && errno == EINTR) {}
    { std::lock_guard<std::mutex> guard(pendingLock); pendingReads.erase(pthread_self()); }
    char drained[16]; bool canceled = false;
    while (::read(wake.fds[0], drained, sizeof(drained)) > 0) canceled = true;
    return !canceled || (fds[0].revents & (POLLIN | POLLHUP | POLLERR)) != 0;
}
}

DWORD GetFileAttributesW(const ShimPathChar* path) {
    struct stat link{}, target{};
    if (lstat(path, &link) != 0) { fail(2); return INVALID_FILE_ATTRIBUTES; }
    if (S_ISLNK(link.st_mode)) return FILE_ATTRIBUTE_REPARSE_POINT | (stat(path, &target) == 0 && S_ISDIR(target.st_mode) ? FILE_ATTRIBUTE_DIRECTORY : 0);
    return S_ISDIR(link.st_mode) ? FILE_ATTRIBUTE_DIRECTORY : FILE_ATTRIBUTE_NORMAL;
}
HANDLE CreateFileW(const ShimPathChar* path, DWORD access, DWORD, void*, DWORD disposition, DWORD, HANDLE) {
    int fd = -1;
    if (disposition == OPEN_EXISTING && access == GENERIC_READ) fd = open(path, O_RDONLY | O_CLOEXEC);
    else if (disposition == CREATE_NEW && access == GENERIC_WRITE) fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    else { fail(ERROR_NOT_SUPPORTED); return INVALID_HANDLE_VALUE; }
    if (fd < 0) { fail(DWORD(errno)); return INVALID_HANDLE_VALUE; }
    return encode(fd);
}
BOOL CreateDirectoryW(const ShimPathChar* path, void*) { if (mkdir(path, 0700) == 0) return TRUE; fail(DWORD(errno)); return FALSE; }
BOOL CloseHandle(HANDLE h) { return close(decode(h)) == 0; }
BOOL GetFileSizeEx(HANDLE h, LARGE_INTEGER* size) {
    struct stat s{}; if (fstat(decode(h), &s) != 0) { fail(DWORD(errno)); return FALSE; }
    size->QuadPart = s.st_size; return TRUE;
}
BOOL ReadFile(HANDLE h, void* buffer, DWORD bytes, DWORD* got, void*) {
    *got = 0; const int fd = decode(h); struct stat s{};
    if (injectedReadFailure(fd)) return FALSE;
    if (fstat(fd, &s) != 0) { fail(readStatus(fd, errno)); return FALSE; }
    const bool pipe = S_ISFIFO(s.st_mode) || S_ISSOCK(s.st_mode);
    if (pipe && !awaitPipe(fd)) { fail(ERROR_OPERATION_ABORTED); return FALSE; }
    for (;;) {
        ssize_t n = read(fd, buffer, bytes);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) { fail(readStatus(fd, errno)); return FALSE; }
        if (n == 0 && pipe && bytes != 0) { fail(ERROR_BROKEN_PIPE); return FALSE; } // The pipe's writer closed.
        *got = DWORD(n); return TRUE; // Zero bytes at end of a file or device, as Win32 does.
    }
}
BOOL CancelSynchronousIo(std::thread::native_handle_type thread) {
    std::lock_guard<std::mutex> guard(pendingLock); auto found = pendingReads.find(thread);
    if (found == pendingReads.end()) { fail(ERROR_NOT_FOUND); return FALSE; }
    const char byte = 1; return write(found->second, &byte, 1) == 1 ? TRUE : FALSE;
}
BOOL WriteFile(HANDLE h, const void* buffer, DWORD bytes, DWORD* wrote, void*) {
    *wrote = 0; auto* in = static_cast<const char*>(buffer);
    while (*wrote < bytes) {
        ssize_t n = write(decode(h), in + *wrote, bytes - *wrote);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { fail(errno == EPIPE ? ERROR_NO_DATA : DWORD(errno)); return FALSE; }
        *wrote += DWORD(n);
    }
    return TRUE;
}
BOOL FlushFileBuffers(HANDLE h) { if (fsync(decode(h)) == 0) return TRUE; fail(DWORD(errno)); return FALSE; }
DWORD GetModuleFileNameW(void*, wchar_t* buffer, DWORD size) {
    char path[4096]; ssize_t n = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (n <= 0) return 0;
    path[n] = 0; auto text = wide(path);
    if (text.size() >= size) return size;
    std::memcpy(buffer, text.c_str(), (text.size() + 1) * sizeof(wchar_t)); return DWORD(text.size());
}
int WideCharToMultiByte(unsigned, DWORD, const wchar_t* text, int count, char* out, int outBytes, const char*, BOOL*) {
    std::string result;
    for (int i = 0; i < count; ++i) {
        uint32_t c = uint32_t(text[i]);
        if ((c >= 0xd800 && c < 0xe000) || c > 0x10ffff) { fail(1113); return 0; }
        if (c < 0x80) result.push_back(char(c));
        else if (c < 0x800) { result.push_back(char(0xc0 | (c >> 6))); result.push_back(char(0x80 | (c & 0x3f))); }
        else if (c < 0x10000) { result.push_back(char(0xe0 | (c >> 12))); result.push_back(char(0x80 | ((c >> 6) & 0x3f))); result.push_back(char(0x80 | (c & 0x3f))); }
        else { result.push_back(char(0xf0 | (c >> 18))); result.push_back(char(0x80 | ((c >> 12) & 0x3f))); result.push_back(char(0x80 | ((c >> 6) & 0x3f))); result.push_back(char(0x80 | (c & 0x3f))); }
    }
    if (!out || !outBytes) return int(result.size());
    if (int(result.size()) > outBytes) { fail(122); return 0; }
    std::memcpy(out, result.data(), result.size()); return int(result.size());
}
DWORD GetLastError() { return lastError; }
void SetLastError(DWORD code) { lastError = code; }
HANDLE GetCurrentProcess() { return reinterpret_cast<HANDLE>(intptr_t(-1)); }
DWORD GetCurrentProcessId() { return DWORD(getpid()); }
HANDLE GetStdHandle(DWORD which) {
    return which == STD_INPUT_HANDLE ? encode(0) : which == STD_OUTPUT_HANDLE ? encode(1) : which == STD_ERROR_HANDLE ? encode(2) : INVALID_HANDLE_VALUE;
}
BOOL IsProcessInJob(HANDLE, HANDLE, BOOL* result) { *result = job().inJob ? TRUE : FALSE; return TRUE; }
BOOL QueryInformationJobObject(HANDLE, JOBOBJECTINFOCLASS kind, void* information, DWORD length, DWORD* returned) {
    auto observed = job();
    if (kind != JobObjectExtendedLimitInformation || length != sizeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION)) { fail(87); return FALSE; }
    std::memcpy(information, &observed.info, sizeof(observed.info)); if (returned) *returned = length; return TRUE;
}
BOOL K32GetProcessMemoryInfo(HANDLE, PROCESS_MEMORY_COUNTERS*, DWORD) { fail(ERROR_NOT_SUPPORTED); return FALSE; }

NTSTATUS BCryptOpenAlgorithmProvider(BCRYPT_ALG_HANDLE* algorithm, const wchar_t*, const wchar_t*, ULONG) { *algorithm = reinterpret_cast<void*>(1); return 0; }
NTSTATUS BCryptGetProperty(void*, const wchar_t*, PUCHAR output, ULONG outputBytes, ULONG* result, ULONG) {
    ULONG size = 1; if (outputBytes != sizeof(size)) return -1; std::memcpy(output, &size, sizeof(size)); *result = sizeof(size); return 0;
}
NTSTATUS BCryptCloseAlgorithmProvider(BCRYPT_ALG_HANDLE, ULONG) { return 0; }
NTSTATUS BCryptCreateHash(BCRYPT_ALG_HANDLE, BCRYPT_HASH_HANDLE* hash, PUCHAR, ULONG, PUCHAR, ULONG, ULONG) { *hash = new chandra::dc::diagnostics::Sha256(); return 0; }
NTSTATUS BCryptDestroyHash(BCRYPT_HASH_HANDLE hash) { delete static_cast<chandra::dc::diagnostics::Sha256*>(hash); return 0; }
NTSTATUS BCryptHashData(BCRYPT_HASH_HANDLE hash, PUCHAR input, ULONG bytes, ULONG) { static_cast<chandra::dc::diagnostics::Sha256*>(hash)->update(input, bytes); return 0; }
NTSTATUS BCryptFinishHash(BCRYPT_HASH_HANDLE hash, PUCHAR output, ULONG bytes, ULONG) {
    if (bytes != 32) return -1;
    auto hex = static_cast<chandra::dc::diagnostics::Sha256*>(hash)->finish();
    for (size_t i = 0; i < 32; ++i) output[i] = UCHAR(std::stoul(hex.substr(i * 2, 2), nullptr, 16));
    return 0;
}

int main(int argc, char** argv) {
    std::signal(SIGPIPE, SIG_IGN); // Win32 WriteFile reports a closed pipe as an error, not a signal.
    std::vector<std::wstring> arguments; std::vector<wchar_t*> pointers;
    for (int i = 0; i < argc; ++i) arguments.push_back(wide(argv[i]));
    for (auto& a : arguments) pointers.push_back(a.data());
    pointers.push_back(nullptr);
    return wmain(argc, pointers.data());
}
