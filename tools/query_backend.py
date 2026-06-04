"""query_backend.py -- send one ExptServer ZMQ verb and print the reply (read-only observation).

Mirrors the REQ framing the monitor/ExptClient use. Decodes the few verbs we use to watch a
shot: get_status (string), get_seq_num / get_num_imgs (8-byte little-endian int). Any other verb
prints the raw reply length.

    <python-with-zmq> query_backend.py get_num_imgs [--url tcp://127.0.0.1:1408]
    <python-with-zmq> query_backend.py get_seq_num
    <python-with-zmq> query_backend.py get_status
"""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verb")
    ap.add_argument("--url", default="tcp://127.0.0.1:1408")
    ap.add_argument("--timeout-ms", type=int, default=3000)
    args = ap.parse_args()

    import zmq
    ctx = zmq.Context()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    try:
        s.connect(args.url)
        s.send_string(args.verb)
        if s.poll(args.timeout_ms) == 0:
            print("%s: TIMEOUT (no reply from %s)" % (args.verb, args.url))
            return
        rep = s.recv()
        if args.verb == "get_status":
            print("%s = %r" % (args.verb, rep.decode("utf-8", "replace")))
        elif args.verb in ("get_seq_num", "get_num_imgs"):
            print("%s = %d" % (args.verb, int.from_bytes(rep, "little")))
        else:
            print("%s -> %d bytes" % (args.verb, len(rep)))
    finally:
        s.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    main()
