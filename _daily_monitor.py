"""Daily-calibration scan monitor: poll queue_list, check loading early, abort at target shots.

Usage: python _daily_monitor.py <expected_label> <target_shots> [<loading_check_at>]
Exits after abort confirmed (or scan finished on its own). Prints file_id + final seq_num.
"""
import sys, os, time, json
import zmq

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

LABEL = sys.argv[1]
TARGET = int(sys.argv[2])
CHECK_AT = int(sys.argv[3]) if len(sys.argv) > 3 else 30

ctx = zmq.Context()

def q(v, t=10000):
    s = ctx.socket(zmq.REQ); s.setsockopt(zmq.LINGER, 0)
    s.connect("tcp://127.0.0.1:1408")
    s.send_string(v)
    r = s.recv() if s.poll(t) else None
    s.close(0)
    return r

def running():
    r = q("queue_list")
    if r is None:
        return "timeout", None
    d = json.loads(r)
    return d.get("running"), d

def loading_from_h5(file_id):
    import h5py, numpy as np
    date = file_id.split("_")[0]
    p = (r"D:\OneDrive - Harvard University\Documents - Yb\Data"
         + "\\" + date + r"\data_" + file_id + r"\data_" + file_id + ".h5")
    try:
        with h5py.File(p, "r") as f:
            l1 = np.array(f["logicals_img1"])
        return float(l1.mean()), l1.shape[0]
    except Exception as e:
        return None, str(e)

def abort():
    for i in range(6):
        q("abort_seq")
        time.sleep(3)
        run, _ = running()
        st = q("get_status")
        if run is None and st and b"stopped" in st:
            return True
    return False

file_id = None
checked_loading = False
stall_seq, stall_t = -1, time.time()
t0 = time.time()
while True:
    run, d = running()
    if run == "timeout":
        print("queue_list timeout (server busy); retrying", flush=True)
        time.sleep(5)
        continue
    if run is None:
        # not running: either finished or not started yet
        if file_id is not None:
            print("SCAN ENDED on its own; file_id=%s" % file_id)
            break
        if time.time() - t0 > 120:
            print("ERROR: scan never started within 120s")
            break
        time.sleep(5)
        continue
    fid = run.get("file_id")
    n = run.get("seq_num") or 0
    if file_id is None and fid:
        file_id = fid
        print("running file_id=%s" % file_id, flush=True)
    # wedge watch: seq_num frozen >120s
    if n != stall_seq:
        stall_seq, stall_t = n, time.time()
    elif time.time() - stall_t > 180:
        print("WEDGE: seq_num frozen at %d for >180s; aborting" % n, flush=True)
        abort()
        break
    if not checked_loading and n >= CHECK_AT and file_id:
        ld, nn = loading_from_h5(file_id)
        if ld is None:
            print("loading check failed: %s (continuing)" % nn, flush=True)
        else:
            print("loading @%s shots = %.3f" % (nn, ld), flush=True)
            if ld < 0.1:
                print("LOADING BAD (<0.1); aborting", flush=True)
                abort()
                break
        checked_loading = True
    if n >= TARGET:
        print("target %d reached (seq_num=%d); aborting" % (TARGET, n), flush=True)
        ok = abort()
        print("abort %s; file_id=%s final_seq_num=%d" % ("confirmed" if ok else "NOT CONFIRMED", file_id, n))
        break
    time.sleep(10)
print("DONE")
