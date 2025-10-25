import argparse
import json
import os
import socket
import struct
import sys
import time
import traceback

import bpy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from blender_render_still import render_single_image  # noqa: E402


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Persistent Blender render daemon.")
    parser.add_argument("--socket", required=True, help="Path to the UNIX domain socket used for job requests.")
    parser.add_argument(
        "--blend",
        required=True,
        help="Path to the base Blender file that should be reloaded before each job.",
    )
    return parser.parse_args(argv)


def recvall(conn, size):
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


def recv_message(conn):
    header = recvall(conn, 4)
    if not header:
        return None
    (length,) = struct.unpack("!I", header)
    payload = recvall(conn, length)
    if not payload:
        return None
    return json.loads(payload.decode("utf-8"))


def send_message(conn, payload):
    data = json.dumps(payload).encode("utf-8")
    header = struct.pack("!I", len(data))
    conn.sendall(header + data)


def handle_job(message, blend_file):
    if "command" in message:
        return {"status": "error", "error": "Unknown command"}, False

    input_path = message["input_path"]
    output_path = message["output_path"]
    resolution = message.get("resolution", 1024)
    camera = message.get("camera", "TopView")
    thread_thickness = message.get("thread_thickness", 0.2)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    metrics = render_single_image(
        input_pes_path=input_path,
        output_image_path=output_path,
        resolution=resolution,
        camera=camera,
        thread_thickness=thread_thickness,
        blend_file=blend_file,
    )

    return {"status": "ok", "metrics": metrics}, False


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = parse_args(argv)
    socket_path = args.socket
    blend_file = args.blend

    if os.path.exists(socket_path):
        os.remove(socket_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    os.chmod(socket_path, 0o660)
    server.listen(5)

    print(f"[daemon] Listening on {socket_path}")

    shutdown_requested = False

    while not shutdown_requested:
        conn, _ = server.accept()
        with conn:
            message = recv_message(conn)
            if not message:
                continue

            if message.get("command") == "shutdown":
                send_message(conn, {"status": "ok", "message": "Shutting down"})
                shutdown_requested = True
                continue

            try:
                start = time.perf_counter()
                response, _ = handle_job(message, blend_file)
                response["elapsed"] = time.perf_counter() - start
                send_message(conn, response)
            except Exception as exc:  # noqa: BLE001
                traceback_str = traceback.format_exc()
                print(f"[daemon] Job failed: {exc}\n{traceback_str}")
                send_message(conn, {"status": "error", "error": str(exc), "traceback": traceback_str})

    server.close()
    if os.path.exists(socket_path):
        os.remove(socket_path)
    print("[daemon] Shutdown complete.")


if __name__ == "__main__":
    main()
