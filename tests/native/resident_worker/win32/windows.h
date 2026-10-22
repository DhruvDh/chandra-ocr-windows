// Test-only, MPL-2.0. Minimal POSIX emulation of the Win32 subset used by inference.cpp,
// inference_core.cpp and worker.cpp so CPU tests compile those Windows sources with g++/clang++
// against a fake Device. Not a Windows SDK: semantics differ (no share modes, symlinks stand in for
// reparse points, Job state comes from CHANDRA_SHIM_JOB). Root still builds with MSVC.
#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <thread>
typedef uint32_t DWORD;
typedef int BOOL;
typedef void* HANDLE;
typedef unsigned char UCHAR;
typedef UCHAR* PUCHAR;
typedef uint32_t ULONG;
typedef int32_t NTSTATUS;
typedef size_t SIZE_T;
typedef uintptr_t ULONG_PTR;
#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif
struct LARGE_INTEGER { int64_t QuadPart; };
#define INVALID_HANDLE_VALUE ((HANDLE)(intptr_t)-1)
#define INVALID_FILE_ATTRIBUTES ((DWORD)-1)
#define FILE_ATTRIBUTE_DIRECTORY 0x10u
#define FILE_ATTRIBUTE_NORMAL 0x80u
#define FILE_ATTRIBUTE_REPARSE_POINT 0x400u
#define GENERIC_READ 0x80000000u
#define GENERIC_WRITE 0x40000000u
#define FILE_SHARE_READ 0x1u
#define CREATE_NEW 1u
#define OPEN_EXISTING 3u
#define FILE_FLAG_SEQUENTIAL_SCAN 0x08000000u
#define CP_UTF8 65001u
#define WC_ERR_INVALID_CHARS 0x80u
#define STD_INPUT_HANDLE ((DWORD)-10)
#define STD_OUTPUT_HANDLE ((DWORD)-11)
#define STD_ERROR_HANDLE ((DWORD)-12)
#define ERROR_INVALID_FUNCTION 1u
#define ERROR_ACCESS_DENIED 5u
#define ERROR_INVALID_HANDLE 6u
#define ERROR_NOT_SUPPORTED 50u
#define ERROR_HANDLE_EOF 38u
#define ERROR_BROKEN_PIPE 109u
#define ERROR_NO_DATA 232u
#define ERROR_PIPE_NOT_CONNECTED 233u
#define ERROR_OPERATION_ABORTED 995u
#define ERROR_NOT_FOUND 1168u
#define JOB_OBJECT_LIMIT_PROCESS_TIME 0x2u
#define JOB_OBJECT_LIMIT_JOB_TIME 0x4u
#define JOB_OBJECT_LIMIT_ACTIVE_PROCESS 0x8u
#define JOB_OBJECT_LIMIT_PROCESS_MEMORY 0x100u
#define JOB_OBJECT_LIMIT_JOB_MEMORY 0x200u
#define JOB_OBJECT_LIMIT_BREAKAWAY_OK 0x800u
#define JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK 0x1000u
#define JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE 0x2000u
enum JOBOBJECTINFOCLASS { JobObjectExtendedLimitInformation = 9 };
struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
    LARGE_INTEGER PerProcessUserTimeLimit, PerJobUserTimeLimit; DWORD LimitFlags;
    SIZE_T MinimumWorkingSetSize, MaximumWorkingSetSize; DWORD ActiveProcessLimit; ULONG_PTR Affinity; DWORD PriorityClass, SchedulingClass;
};
struct IO_COUNTERS { uint64_t ReadOperationCount, WriteOperationCount, OtherOperationCount, ReadTransferCount, WriteTransferCount, OtherTransferCount; };
struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
    JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation; IO_COUNTERS IoInfo;
    SIZE_T ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemoryUsed, PeakJobMemoryUsed;
};
using ShimPathChar = std::filesystem::path::value_type; // char on POSIX, where path::c_str() is narrow.
DWORD GetFileAttributesW(const ShimPathChar* path);
HANDLE CreateFileW(const ShimPathChar* path, DWORD access, DWORD share, void* security, DWORD disposition, DWORD flags, HANDLE templateFile);
BOOL CreateDirectoryW(const ShimPathChar* path, void* security);
BOOL CloseHandle(HANDLE);
BOOL GetFileSizeEx(HANDLE, LARGE_INTEGER*);
BOOL ReadFile(HANDLE, void* buffer, DWORD bytes, DWORD* read, void* overlapped);
BOOL CancelSynchronousIo(std::thread::native_handle_type thread); // HANDLE under MSVC, pthread_t here.
BOOL WriteFile(HANDLE, const void* buffer, DWORD bytes, DWORD* written, void* overlapped);
BOOL FlushFileBuffers(HANDLE);
DWORD GetModuleFileNameW(void* module, wchar_t* buffer, DWORD size);
int WideCharToMultiByte(unsigned codePage, DWORD flags, const wchar_t* wide, int wideCount, char* out, int outBytes, const char* defaultChar, BOOL* usedDefault);
DWORD GetLastError();
void SetLastError(DWORD);
HANDLE GetCurrentProcess();
DWORD GetCurrentProcessId();
HANDLE GetStdHandle(DWORD);
BOOL IsProcessInJob(HANDLE process, HANDLE job, BOOL* result);
BOOL QueryInformationJobObject(HANDLE job, JOBOBJECTINFOCLASS kind, void* information, DWORD length, DWORD* returned);
int wmain(int argc, wchar_t** argv);
