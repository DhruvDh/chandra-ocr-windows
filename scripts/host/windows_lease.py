import base64, ctypes, hmac, json, socket, subprocess, sys, threading, time
from ctypes import wintypes as W
K = ctypes.WinDLL('kernel32', use_last_error=True)
class IO(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
class BASIC(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit',ctypes.c_longlong),('PerJobUserTimeLimit',ctypes.c_longlong),('LimitFlags',W.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),('MaximumWorkingSetSize',ctypes.c_size_t),('ActiveProcessLimit',W.DWORD),('Affinity',ctypes.c_size_t),('PriorityClass',W.DWORD),('SchedulingClass',W.DWORD)]
class LIMIT(ctypes.Structure):
    _fields_ = [('BasicLimitInformation',BASIC),('IoInfo',IO),('ProcessMemoryLimit',ctypes.c_size_t),('JobMemoryLimit',ctypes.c_size_t),('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]
class START(ctypes.Structure):
    _fields_ = [('cb',W.DWORD),('lpReserved',W.LPWSTR),('lpDesktop',W.LPWSTR),('lpTitle',W.LPWSTR),('dwX',W.DWORD),('dwY',W.DWORD),('dwXSize',W.DWORD),('dwYSize',W.DWORD),('dwXCountChars',W.DWORD),('dwYCountChars',W.DWORD),('dwFillAttribute',W.DWORD),('dwFlags',W.DWORD),('wShowWindow',W.WORD),('cbReserved2',W.WORD),('lpReserved2',ctypes.POINTER(ctypes.c_byte)),('hStdInput',W.HANDLE),('hStdOutput',W.HANDLE),('hStdError',W.HANDLE)]
class PROCESS(ctypes.Structure):
    _fields_ = [('hProcess',W.HANDLE),('hThread',W.HANDLE),('dwProcessId',W.DWORD),('dwThreadId',W.DWORD)]
K.CreateJobObjectW.argtypes = [ctypes.c_void_p,W.LPCWSTR]; K.CreateJobObjectW.restype = W.HANDLE
K.SetInformationJobObject.argtypes = [W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD]; K.SetInformationJobObject.restype = W.BOOL
K.CreateProcessW.argtypes = [W.LPCWSTR,W.LPWSTR,ctypes.c_void_p,ctypes.c_void_p,W.BOOL,W.DWORD,ctypes.c_void_p,W.LPCWSTR,ctypes.POINTER(START),ctypes.POINTER(PROCESS)]; K.CreateProcessW.restype = W.BOOL
K.AssignProcessToJobObject.argtypes = [W.HANDLE,W.HANDLE]; K.AssignProcessToJobObject.restype = W.BOOL
K.ResumeThread.argtypes = [W.HANDLE]; K.ResumeThread.restype = W.DWORD
K.TerminateProcess.argtypes = [W.HANDLE,W.UINT]; K.TerminateProcess.restype = W.BOOL
K.TerminateJobObject.argtypes = [W.HANDLE,W.UINT]; K.TerminateJobObject.restype = W.BOOL
K.WaitForSingleObject.argtypes = [W.HANDLE,W.DWORD]; K.WaitForSingleObject.restype = W.DWORD
K.GetExitCodeProcess.argtypes = [W.HANDLE,ctypes.POINTER(W.DWORD)]; K.GetExitCodeProcess.restype = W.BOOL
K.CloseHandle.argtypes = [W.HANDLE]; K.CloseHandle.restype = W.BOOL
K.GetStdHandle.argtypes = [W.DWORD]; K.GetStdHandle.restype = W.HANDLE
config = json.loads(base64.b64decode(sys.argv[1]))
lease_seconds = config.get('lease_seconds', 15)
listener = socket.socket()
listener.bind(('127.0.0.1', config['lease_port']))
listener.listen(4)
listener.settimeout(.2)
job = K.CreateJobObjectW(None, None)
if not job: raise ctypes.WinError(ctypes.get_last_error())
process = PROCESS()
try:
    limits = LIMIT(); limits.BasicLimitInformation.LimitFlags = 0x2000 # KILL_ON_JOB_CLOSE
    if not K.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)): raise ctypes.WinError(ctypes.get_last_error())
    startup = START(); startup.cb = ctypes.sizeof(startup); startup.dwFlags = 0x100
    startup.hStdInput = K.GetStdHandle(-10); startup.hStdOutput = K.GetStdHandle(-11); startup.hStdError = K.GetStdHandle(-12)
    command = ctypes.create_unicode_buffer(subprocess.list2cmdline(config['argv']))
    # The child cannot execute or spawn descendants before ownership is assigned.
    if not K.CreateProcessW(None, command, None, None, True, 0x4, None, config['cwd'], ctypes.byref(startup), ctypes.byref(process)): raise ctypes.WinError(ctypes.get_last_error())
    if not K.AssignProcessToJobObject(job, process.hProcess):
        K.TerminateProcess(process.hProcess, 1)
        K.WaitForSingleObject(process.hProcess, 5000)
        raise ctypes.WinError(ctypes.get_last_error())
    if K.ResumeThread(process.hThread) == 0xffffffff: raise ctypes.WinError(ctypes.get_last_error())
    lease = [time.monotonic()]
    disconnected = threading.Event()
    def heartbeat():
        connection = None
        try:
            while time.monotonic() - lease[0] <= lease_seconds:
                try:
                    connection, _ = listener.accept()
                    connection.settimeout(.2)
                    break
                except socket.timeout:
                    continue
            if connection is None:
                return
            buffer = b''
            authenticated = False
            while True:
                try:
                    data = connection.recv(4096)
                except socket.timeout:
                    continue
                if not data:
                    disconnected.set()
                    return
                buffer += data
                if len(buffer) > 8192:
                    disconnected.set()
                    return
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    if not authenticated:
                        if not hmac.compare_digest(line, config['lease_token'].encode()):
                            disconnected.set()
                            return
                        authenticated = True
                        connection.sendall(b"owned\n")
                    elif line != b'alive':
                        disconnected.set()
                        return
                    lease[0] = time.monotonic()
        finally:
            if connection is not None:
                connection.close()
    threading.Thread(target=heartbeat, daemon=True).start()
    while K.WaitForSingleObject(process.hProcess, 100) == 0x102:
        if disconnected.is_set() or time.monotonic() - lease[0] > lease_seconds:
            print("Owned worker lease ended: " + ("lease EOF" if disconnected.is_set() else "heartbeat deadline"), file=sys.stderr, flush=True)
            K.TerminateJobObject(job, 1)
            if K.WaitForSingleObject(process.hProcess, 5000) != 0: raise RuntimeError('Owned worker did not stop within deadline')
            sys.exit(1)
    code = W.DWORD()
    if not K.GetExitCodeProcess(process.hProcess, ctypes.byref(code)): raise ctypes.WinError(ctypes.get_last_error())
    sys.exit(code.value)
finally:
    # Unnamed, non-inheritable job handle; only this lease owns its descendants.
    listener.close()
    K.CloseHandle(job)
    if process.hThread: K.CloseHandle(process.hThread)
    if process.hProcess: K.CloseHandle(process.hProcess)
