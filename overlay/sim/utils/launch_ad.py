import subprocess
import time
import os
import signal
import errno
import pickle
import select


class PlannerIPCError(RuntimeError):
    """Raised when the external planner exits or stalls during FIFO exchange."""


def _check_planner(process):
    if process is None:
        return
    return_code = process.poll()
    if return_code is not None:
        raise PlannerIPCError(f"planner exited with return code {return_code}")


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise PlannerIPCError("planner FIFO exchange timed out")
    return remaining


def fifo_send(path, value, process, timeout_seconds=120.0):
    """Write one framed-by-close pickle without blocking on a dead reader."""

    payload = memoryview(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    deadline = time.monotonic() + float(timeout_seconds)
    descriptor = None
    while descriptor is None:
        _check_planner(process)
        _remaining(deadline)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as error:
            if error.errno != errno.ENXIO:
                raise
            time.sleep(0.02)
    try:
        while payload:
            _check_planner(process)
            wait = min(0.1, _remaining(deadline))
            _, writable, exceptional = select.select([], [descriptor], [descriptor], wait)
            if exceptional:
                raise PlannerIPCError("planner FIFO became exceptional while writing")
            if not writable:
                continue
            try:
                written = os.write(descriptor, payload)
            except BrokenPipeError as error:
                raise PlannerIPCError("planner closed observation FIFO") from error
            payload = payload[written:]
    finally:
        os.close(descriptor)


def fifo_receive(path, process, timeout_seconds=120.0):
    """Read one framed-by-close pickle without blocking on a dead writer."""

    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        _check_planner(process)
        _remaining(deadline)
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        chunks = []
        try:
            while True:
                _check_planner(process)
                wait = min(0.1, _remaining(deadline))
                readable, _, exceptional = select.select([descriptor], [], [descriptor], wait)
                if exceptional:
                    raise PlannerIPCError("planner FIFO became exceptional while reading")
                if not readable:
                    continue
                try:
                    chunk = os.read(descriptor, 1 << 20)
                except BlockingIOError:
                    continue
                if chunk:
                    chunks.append(chunk)
                    continue
                if chunks:
                    try:
                        return pickle.loads(b"".join(chunks))
                    except Exception as error:
                        raise PlannerIPCError(
                            f"planner FIFO returned an invalid pickle: {error!r}"
                        ) from error
                break
        finally:
            os.close(descriptor)
        time.sleep(0.02)


def launch(shell_path, cuda_id, output):
    os.makedirs(output, exist_ok=True)
    print(os.path.join(output, 'output.txt'))
    print(shell_path, cuda_id, output)
    with open(os.path.join(output, 'output.txt'), 'w') as f:
        # The bundled launchers use bash syntax and declare a bash shebang.
        # Invoke them directly so the simulator does not depend on zsh being
        # installed on the benchmark host.
        process = subprocess.Popen(
            [shell_path, cuda_id, output],
            stdout=f,
            stderr=f,
            # The conda launcher may create one or more child processes.  A
            # dedicated session lets the simulator reap the whole planner
            # tree after a crash instead of leaving an orphan FIFO reader.
            start_new_session=True,
        )
    return process


def terminate(process, grace_seconds=5):
    """Terminate the complete planner process group, including conda children."""

    if process.poll() is not None:
        return process.returncode
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return process.poll()
    try:
        return process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()


def check_alive(process, tolerant=100):
    i = 0
    while i < tolerant:
        return_code = process.poll()
        if return_code is not None:
            print(f"The AD algorithm completed with return code {return_code}.")
            return return_code
        elif i % 5 == 0:
            print(f"The AD algorithm is still running, remaining tolerant {tolerant - i}.")
        time.sleep(1)
        i += 1
    terminate(process)
    print("The AD algorithm process is killed.")
    return process.returncode
