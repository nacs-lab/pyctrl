"""Subprocess helper for the control_channel two-process test.

Sends a scripted sequence of ExptServer control verbs over a real ZMQ REQ socket, so the test
exercises the genuine cross-process path (a separate OS process issuing pause/abort/start verbs
to the bound ExptServer, handled by its worker thread concurrently with the run-loop gate).

Usage: ``python control_writer_helper.py <url> "<delay>:<verb>,<delay>:<verb>,..."``
Each step sleeps ``delay`` seconds (relative to the previous step) then sends ``verb`` (e.g.
``pause_seq`` / ``abort_seq`` / ``start_seq``) on a fresh REQ socket (avoids REQ FSM lockups).
"""

import sys
import time

import zmq


def main():
    url = sys.argv[1]
    script = sys.argv[2] if len(sys.argv) > 2 else ""
    ctx = zmq.Context()
    try:
        for item in script.split(","):
            item = item.strip()
            if not item:
                continue
            delay, verb = item.split(":")
            time.sleep(float(delay))
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(url)
            try:
                sock.send_string(verb)
                if sock.poll(2000):
                    sock.recv()
            finally:
                sock.close(linger=0)
    finally:
        ctx.term()


if __name__ == "__main__":
    main()
