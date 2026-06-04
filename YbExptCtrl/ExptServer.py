"""ExptServer.py -- the ZMQ hub hosted by whichever backend is live.

VERBATIM COPY of ``matlab_new/YbExptCtrl/ExptServer.py`` (do not diverge the body):
the ExptServer ZMQ protocol is the SINGLE cross-backend contract shared by the MATLAB
runner, the pyctrl run loop, and the yb_analysis monitor (references/runtime-design.md:
"one contract ... hosted by whichever backend is live"). In scenario 3 (new monitor +
pyctrl, MATLAB off) the pyctrl run loop (``runner.py``) instantiates this and drains the
queue via ``pop_next_descriptor`` / ``pop_next_job``; the wire format (verbs, the
``get_imgs`` column-major/shape-prefix/0-separated flat-double format ``_process_imgs``
parses, ``ping``->``pong``) must stay byte-identical to the MATLAB-hosted copy or the
monitor mis-reads it. Keep this file a faithful copy and re-sync if the MATLAB one changes.

Scenario-3 note: only TWO submission paths exist today -- a JSON ``submit_scan_descriptor``
(the new monitor) and the MATLAB ".m run-button" ``submit_job`` (a MATLAB byte-stream
payload via ybStartScan). The latter needs a live MATLAB and is unavailable under pyctrl,
so the pyctrl runner consumes the descriptor path only (it dispatches a descriptor into a
JSON job payload it produces and consumes itself -- see runner.py).
"""

import errno
import json
import os
import tempfile
import threading
import time
import zmq
from collections import deque
from enum import Enum
import array

QUEUE_PATH = os.path.join(tempfile.gettempdir(), 'nacsctl', 'runner_queue.json')
HISTORY_CAP = 50


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


class ExptServer(object):
    class State(Enum):
        Init = 0
        Paused = 1
        Running = 2
    class WorkerRequest(Enum):
        NoRequest = 0
        Stop = 1
    class SeqRequest(Enum):
        NoRequest = 0
        Pause = 1
        Abort = 2

    def recreate_sock(self):
        if self.__sock is not None:
            try:
                self.__sock.close(linger=0)
            except Exception:
                pass
            self.__sock = None
        sock = self.__ctx.socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.IMMEDIATE, 1)
        last_err = None
        bound = False
        try:
            for attempt in range(10):
                try:
                    sock.bind(self.__url)
                    bound = True
                    self.__sock = sock
                    return
                except zmq.ZMQError as e:
                    last_err = e
                    if e.errno in (errno.EADDRINUSE, zmq.EADDRINUSE):
                        time.sleep(0.2)
                        continue
                    raise
        finally:
            if not bound:
                try:
                    sock.close(linger=0)
                except Exception:
                    pass
        raise RuntimeError(
            f"Could not bind {self.__url} after 10 attempts "
            f"(port likely held by a stale process; run kill_port "
            f"from yb_analysis.acquisition.port_utils)") from last_err

    def __init__(self, url: str):
        # network
        self.__url = url
        self.__ctx = zmq.Context()
        self.__sock = None
        self.recreate_sock()
        self.timeout = 500

        # lock whenever accessing or changing state variables
        self.__data_lock = threading.Lock()
        # lock for worker request
        self.__worker_lock = threading.Lock()
        # lock for seq request
        self.__seq_lock = threading.Lock()
        # lock for expt imgs
        self.__expt_lock = threading.Lock()
        # lock for the job queue (NEW)
        self.__queue_lock = threading.Lock()

        # worker. This worker will handle network requests
        with self.__worker_lock:
            self.__worker_req = self.WorkerRequest.NoRequest
        self.__worker = threading.Thread(target=self.__worker_func)
        self.__worker.start()

        # data variables
        with self.__expt_lock:
            self.expt_imgs = deque()  # this deque is the one the expt thread uses.
        with self.__data_lock:
            self.imgs = deque()
            self.nseq = 0
            self.dateStamp = ""
            self.timeStamp = ""
            self.nseq_imgs = 0  # number of sequences of images stored
        self.temp_imgs = []  # stored mid sequence

        # status of seq
        with self.__seq_lock:
            self.__seq_req = self.SeqRequest.NoRequest
            # reached-paused ack (pyctrl backend; mirrors runSeq2's IsPausedRunSeq). The coarse
            # get_status reflects the pause REQUEST; this flag is the run loop's report that it
            # has actually parked. See ack_paused / is_paused.
            self.__is_paused = False
        with self.__data_lock:
            self.seq_status = self.State.Init

        # camera state (managed by SequenceRunner; queried/commanded via ZMQ)
        self.__camera_lock = threading.Lock()
        self._camera = {
            'connected': False,
            'roi': [0, 0, 4096, 2304],
            'exposure_time': 0.1,
            'error': '',
            # Extended Orca telemetry (pyctrl backend; MATLAB leaves these at defaults).
            # Surfaced on the monitor + web Camera card. set_camera_status() refreshes them.
            'trigger': '',          # 'external (rising)' / 'internal'
            'cooler': '',           # cooler mode: off / on / max
            'cooler_status': '',    # cooler state readout (e.g. 'ready')
            'temperature': None,    # sensor temperature, degrees C
        }
        self._camera_pending = None  # {'cmd': str, 'roi': list, 'exposure_time': float} or None

        # dummy-sequence keep-alive mode. SequenceRunner polls this every
        # idle iteration:
        #   'off'     -> short pause, no hardware activity
        #   'default' -> run the pre-compiled DummySeq
        #   'last'    -> run the cached last successful real seq; if no
        #               cache yet, fall back to default and surface that
        #               via __last_fallback_active.
        # Cached-seq frames flow through the normal store_imgs/seq_finish
        # path but with scan_id=-1 (set by SequenceRunner before replay),
        # so the Python control loop routes them to display-only.
        self.__dummy_lock = threading.Lock()
        self.__dummy_mode = 'last'
        # metadata about the last seq cached on the MATLAB side
        self.__last_seq_meta = {
            'available': False,
            'name': '',
            'file_id': '',
            'captured_at': None,
        }
        self.__last_fallback_active = False

        # job queue (NEW)
        with self.__queue_lock:
            self.__next_job_id = 1
            self.__queue = []   # list of {id, seqName, scangroupDump (bytes b64 on disk), state, enqueued_ts, start_ts, finish_ts, status}
            self.__history = []
            self.__load_queue()

    def __del__(self):
        self.stop_worker()
        try:
            self.__sock.close(linger=0)
        except Exception:
            pass
        try:
            self.__ctx.destroy(linger=0)
        except Exception:
            pass

    def reset(self):
        self.stop_worker()
        self.recreate_sock()
        with self.__expt_lock:
            self.expt_imgs = deque()
        with self.__data_lock:
            self.imgs = deque()
            self.nseq = 0
            self.dateStamp = ""
            self.timeStamp = ""
            self.nseq_imgs = 0
        with self.__seq_lock:
            self.__seq_req = self.SeqRequest.NoRequest
            self.__is_paused = False
        with self.__data_lock:
            self.seq_status = self.State.Init
        self.start_worker()

    def stop_worker(self):
        if hasattr(self, '_ExptServer__worker'):
            with self.__worker_lock:
                self.__worker_req = self.WorkerRequest.Stop
            self.__worker.join()
        else:
            return

    def start_worker(self):
        if hasattr(self, '_ExptServer__worker'):
            if self.__worker.is_alive():
                return
        with self.__worker_lock:
            self.__worker_req = self.WorkerRequest.NoRequest
        self.__worker = threading.Thread(target=self.__worker_func)
        self.__worker.start()

    def handle_msg(self, addr, msg_str: str) -> bool:
        # Method to handle different requests from external clients
        if msg_str == "pause_seq":
            rep = self.pause_seq()
            self.safe_send_string(addr, rep)
        elif msg_str == "abort_seq":
            rep = self.abort_seq()
            self.safe_send_string(addr, rep)
        elif msg_str == "start_seq":
            rep = self.start_seq_serv()
            self.safe_send_string(addr, rep)
        elif msg_str == "get_status":
            rep = self.get_status()
            self.safe_send_string(addr, rep)
        elif msg_str == "get_imgs":
            rep = self.get_imgs()
            self.safe_send(addr, rep)
        elif msg_str == "get_seq_num":
            rep = self.get_seq_num()
            self.safe_send(addr, rep.to_bytes(8, 'little'))
        elif msg_str == "get_num_imgs":
            rep = self.get_num_imgs()
            self.safe_send(addr, rep.to_bytes(8, 'little'))
        elif msg_str == "get_config":
            datestr, timestr = self.get_config()
            self.safe_send_string(addr, datestr, zmq.SNDMORE)
            self.__sock.send_string(timestr)
        elif msg_str == "ping":
            self.safe_send_string(addr, "pong")
        elif msg_str == "submit_job":
            # additional frames: payload bytes, (optional) summary JSON string
            payload = self.safe_recv()
            if payload is None:
                self.safe_send_string(addr, "error: missing payload")
                return True
            summary_json = self.safe_recv_string()  # may be None or ''
            summary = None
            if summary_json:
                try:
                    summary = json.loads(summary_json)
                except Exception as ex:
                    # Log a short preview of the bad JSON to help triage.
                    preview = summary_json[:200]
                    print(f"[ExptServer] warning: could not parse summary JSON ({ex}); "
                          f"discarding. preview={preview!r}")
            job_id = self.submit_job(payload, summary=summary)
            self.safe_send(addr, job_id.to_bytes(8, 'little'))
        elif msg_str == "queue_list":
            js = json.dumps(self.queue_list()).encode('utf-8')
            self.safe_send(addr, js)
        elif msg_str == "queue_remove":
            raw = self.safe_recv()
            if raw is None:
                self.safe_send_string(addr, "error: missing id")
                return True
            jid = int.from_bytes(raw, 'little', signed=False)
            ok = self.queue_remove(jid)
            self.safe_send_string(addr, "ok" if ok else "error: not queued")
        elif msg_str == "queue_move":
            raw_id = self.safe_recv()
            raw_dir = self.safe_recv_string()
            if raw_id is None or raw_dir is None:
                self.safe_send_string(addr, "error: missing args")
                return True
            jid = int.from_bytes(raw_id, 'little', signed=False)
            ok = self.queue_move(jid, raw_dir)
            self.safe_send_string(addr, "ok" if ok else "error: cannot move")
        elif msg_str == "submit_scan_descriptor":
            # Frames: descriptor JSON string, (optional) label string
            desc_json = self.safe_recv_string()
            if desc_json is None:
                self.safe_send_string(addr, "error: missing descriptor")
                return True
            label = self.safe_recv_string() or ''
            did = self.submit_scan_descriptor(desc_json, label=label)
            self.safe_send(addr, did.to_bytes(8, 'little'))
        elif msg_str == "descriptor_remove":
            raw = self.safe_recv()
            if raw is None:
                self.safe_send_string(addr, "error: missing id")
                return True
            did = int.from_bytes(raw, 'little', signed=False)
            ok = self.descriptor_remove(did)
            self.safe_send_string(addr, "ok" if ok else "error: not queued")
        elif msg_str == "camera_init":
            # Payload: JSON list [roi] (legacy) or dict {roi, exposure_time}
            payload_str = self.safe_recv_string()
            roi, exposure = self.__parse_camera_payload(payload_str)
            with self.__camera_lock:
                self._camera_pending = {
                    'cmd': 'init',
                    'roi': roi,
                    'exposure_time': exposure,
                }
            self.safe_send_string(addr, "ok")
        elif msg_str == "camera_apply_settings":
            payload_str = self.safe_recv_string()
            roi, exposure = self.__parse_camera_payload(payload_str)
            with self.__camera_lock:
                self._camera_pending = {
                    'cmd': 'apply_settings',
                    'roi': roi,
                    'exposure_time': exposure,
                }
            self.safe_send_string(addr, "ok")
        elif msg_str == "camera_close":
            with self.__camera_lock:
                self._camera_pending = {'cmd': 'close', 'roi': [0, 0, 0, 0]}
            self.safe_send_string(addr, "ok")
        elif msg_str == "camera_status":
            with self.__camera_lock:
                status = dict(self._camera)
            self.safe_send(addr, json.dumps(status).encode('utf-8'))
        elif msg_str == "set_dummy_enabled":
            # Legacy verb kept for backward compatibility: bool -> mode.
            val_str = self.safe_recv_string() or ''
            enabled = val_str.strip().lower() in ('1', 'true', 'yes', 'on')
            with self.__dummy_lock:
                self.__dummy_mode = 'default' if enabled else 'off'
            self.safe_send_string(addr, "ok")
        elif msg_str == "get_dummy_enabled":
            with self.__dummy_lock:
                val = '1' if self.__dummy_mode != 'off' else '0'
            self.safe_send_string(addr, val)
        elif msg_str == "set_dummy_mode":
            val_str = (self.safe_recv_string() or '').strip().lower()
            if val_str not in ('off', 'default', 'last'):
                self.safe_send_string(addr, f"error: bad mode {val_str!r}")
                return True
            with self.__dummy_lock:
                self.__dummy_mode = val_str
            self.safe_send_string(addr, "ok")
        elif msg_str == "get_dummy_mode":
            with self.__dummy_lock:
                val = self.__dummy_mode
            self.safe_send_string(addr, val)
        elif msg_str == "set_last_seq_meta":
            # MATLAB pushes {name, file_id} after caching a seq. captured_at
            # is stamped server-side so wall-clock skew between MATLAB and
            # Python doesn't matter.
            payload_str = self.safe_recv_string() or ''
            try:
                data = json.loads(payload_str) if payload_str else {}
            except Exception:
                data = {}
            with self.__dummy_lock:
                self.__last_seq_meta = {
                    'available': True,
                    'name': str(data.get('name', '')),
                    'file_id': str(data.get('file_id', '')),
                    'captured_at': time.time(),
                }
            self.safe_send_string(addr, "ok")
        elif msg_str == "clear_last_seq_meta":
            with self.__dummy_lock:
                self.__last_seq_meta = {
                    'available': False, 'name': '', 'file_id': '',
                    'captured_at': None,
                }
                self.__last_fallback_active = False
            self.safe_send_string(addr, "ok")
        elif msg_str == "set_last_fallback":
            val_str = (self.safe_recv_string() or '').strip().lower()
            active = val_str in ('1', 'true', 'yes', 'on')
            with self.__dummy_lock:
                self.__last_fallback_active = active
            self.safe_send_string(addr, "ok")
        elif msg_str == "last_seq_status":
            with self.__dummy_lock:
                status = dict(self.__last_seq_meta)
                status['fallback_active'] = self.__last_fallback_active
                status['mode'] = self.__dummy_mode
            self.safe_send(addr, json.dumps(status).encode('utf-8'))
        else:
            self.safe_send_string(addr, f'')
            return False
        return True

    def safe_receive(func):
        def f(self):
            try:
                msg = func(self)
            except Exception:
                msg = None
            return msg
        return f

    @safe_receive
    def safe_recv(self):
        return self.__sock.recv(zmq.NOBLOCK)

    @safe_receive
    def safe_recv_string(self):
        return self.__sock.recv_string(zmq.NOBLOCK)

    def finish_recv(func):
        def f(self, *args, **kwargs):
            # finish receiving messages
            msg = self.safe_recv()
            while msg is not None:
                msg = self.safe_recv()
            func(self, *args, **kwargs)
        return f

    @finish_recv
    def safe_send_string(self, addr, msg_str, flag=0):
        # send reply
        self.__sock.send(addr, zmq.SNDMORE)
        self.__sock.send(b'', zmq.SNDMORE)
        self.__sock.send_string(msg_str, flag)

    @finish_recv
    def safe_send(self, addr, msg, flag=0):
        # send reply
        self.__sock.send(addr, zmq.SNDMORE)
        self.__sock.send(b'', zmq.SNDMORE)
        self.__sock.send(msg, flag)

    def __check_worker_req(self):
        with self.__worker_lock:
            return self.__worker_req

    def __worker_func(self):
        # worker function
        while self.__check_worker_req() != self.WorkerRequest.Stop:
            try:
                if self.__sock.poll(self.timeout) == 0:  # in milliseconds
                    continue
                addr = self.safe_recv()
                delimit = self.safe_recv_string()
                msg_str = self.safe_recv_string()
                if msg_str is None:
                    self.safe_send_string(addr, "Send more")
                    continue
                self.handle_msg(addr, msg_str)
            except Exception as ex:
                # A single bad message must NOT kill the worker — if it does,
                # every subsequent REQ (images, status, queue) will silently
                # time out and the client sees nothing, with no error surfaced.
                import traceback
                print(f"[ExptServer] worker caught {type(ex).__name__}: {ex}")
                traceback.print_exc()
        print("Worker finishing")

    # functions for either thread but mostly for the msg handler
    def pause_seq(self) -> str:
        with self.__data_lock:
            if self.seq_status == self.State.Running:
                self.seq_status = self.State.Paused
                with self.__seq_lock:
                    self.__seq_req = self.SeqRequest.Pause
                    self.__is_paused = False    # requested, not yet reached (runner acks on park)
                res = "Sequence Paused"
            else:
                res = "Sequence is not running"
        return res

    def abort_seq(self) -> str:
        with self.__data_lock:
            if self.seq_status == self.State.Running or self.seq_status == self.State.Paused:
                self.seq_status = self.State.Init
                with self.__seq_lock:
                    self.__seq_req = self.SeqRequest.Abort
                    self.__is_paused = False    # abort un-parks
                res = "Sequence Aborted"
            else:
                res = "Sequence is not running"
        return res

    def get_status(self) -> str:
        with self.__data_lock:
            if self.seq_status == self.State.Init:
                res = "Sequence is stopped"
            elif self.seq_status == self.State.Paused:
                res = "Sequence is paused"
            elif self.seq_status == self.State.Running:
                res = "Sequence is running"
            else:
                res = "Sequence status is unknown"
        return res

    def pop_img(self):
        with self.__data_lock:
            try:
                res = self.imgs.pop()
            except Exception:
                res = None
        return res

    def get_seq_num(self) -> int:
        with self.__data_lock:
            return self.nseq

    def get_num_imgs(self) -> int:
        with self.__data_lock:
            return self.nseq_imgs

    def get_config(self):
        with self.__data_lock:
            return self.dateStamp, self.timeStamp

    def get_imgs(self):
        # returns bytes to be sent across the network
        zero_array = array.array('d', [0])
        res = bytearray()
        # Atomically: read nseqs, swap deques, reset counter. Reading nseqs
        # outside the lock and swapping later is a race — MATLAB can call
        # seq_finish between the two, leaving `self.imgs` with more sequences
        # than `nseqs` says. Subsequent drains then underflow, pop_img returns
        # None, .tobytes() raises, and the worker thread dies silently.
        with self.__data_lock:
            with self.__expt_lock:
                nseqs = self.nseq_imgs
                self.expt_imgs, self.imgs = self.imgs, self.expt_imgs
                self.nseq_imgs = 0
        nseqs_array = array.array('d', [nseqs])
        res.extend(nseqs_array.tobytes())
        n_transfer = 0
        while n_transfer < nseqs:
            next_img = self.pop_img()
            while next_img is not None and next_img != b'':
                res.extend(next_img.tobytes())
                next_img = self.pop_img()
            res.extend(zero_array.tobytes())
            n_transfer += 1
            if next_img is None:
                # Deque exhausted before we hit `nseqs` sentinels — shouldn't
                # happen now that nseqs and the swap are atomic, but bail
                # cleanly instead of raising if it ever does.
                break
        return res

    # this one is only for msg handler
    def start_seq_serv(self) -> str:
        with self.__data_lock:
            if self.seq_status == self.State.Paused:
                with self.__seq_lock:
                    self.__seq_req = self.SeqRequest.NoRequest
                    self.__is_paused = False    # resume un-parks
                self.seq_status = self.State.Running
                res = "Sequence should now be running"
            else:
                res = "Sequence was not in Paused state. To start a new sequence, use the main MATLAB instance"
        return res

    # functions for the Expt thread
    def check_request(self):
        with self.__seq_lock:
            res = self.__seq_req.value  # get value for Matlab
        return res

    def ack_paused(self, parked):
        """Record whether the run loop has ACTUALLY parked at the per-sequence gate (pyctrl
        backend; reached-paused ack).

        Request-vs-reached: :meth:`pause_seq` flips the coarse State to Paused on the bare pause
        REQUEST (before the in-flight FPGA shot finishes -- a lie mid-shot). This ack is set True
        by the runner only once it truly parks in the pause spin, and False when it resumes or
        aborts (``control_channel.check_pause_abort`` calls it). It is the reached-paused truth
        ``runSeq2`` keeps as ``IsPausedRunSeq`` -- consumed by the run-loop coherency tests and the
        deferred ``get_status_rich`` verb. The MATLAB hub never calls it (the coarse path stands)."""
        with self.__seq_lock:
            self.__is_paused = bool(parked)

    def is_paused(self):
        """True iff the run loop has reported (via :meth:`ack_paused`) that it is actually parked
        at the gate -- distinct from :meth:`get_status`, which reflects the pause REQUEST."""
        with self.__seq_lock:
            return self.__is_paused

    def clear_seq_request(self):
        """Reset the seq request to NoRequest without touching ``seq_status`` (pyctrl backend).

        Used by the idle/keep-alive loop to CONSUME an abort that arrived while no scan was
        running: there is nothing to abort during idle, so the request is cleared (the keep-alive
        is silenced for that iteration by the IdleScheduler abort gate) instead of lingering and
        suppressing the dummy keep-alive indefinitely. Real scans still clear at job start via
        :meth:`start_scan` (clear-at-job-start); this is the idle-only consumer."""
        with self.__seq_lock:
            self.__seq_req = self.SeqRequest.NoRequest
            self.__is_paused = False

    def start_scan(self):
        with self.__data_lock:
            self.seq_status = self.State.Running
        with self.__seq_lock:
            self.__seq_req = self.SeqRequest.NoRequest
        scan_id = round(time.time() * 1000)
        return scan_id

    def store_imgs(self, data, scan_id=-1, seq_id=-1):
        # MATLAB passes `data` as a lazy matlab.double wrapper around its own
        # memory. Force an eager copy into Python-owned bytes here — once the
        # MATLAB sequence completes, the underlying buffer can go stale and
        # later .tobytes() in get_imgs returns garbage or raises silently.
        data = array.array('d', data)
        if not self.temp_imgs:
            self.temp_imgs.append(array.array('d', [scan_id]))
            self.temp_imgs.append(array.array('d', [seq_id]))
        self.temp_imgs.append(data)

    def seq_finish(self):
        with self.__data_lock:
            self.nseq = self.nseq + 1
            self.nseq_imgs = self.nseq_imgs + 1
            with self.__expt_lock:
                for data in self.temp_imgs:
                    self.expt_imgs.appendleft(data)
                self.expt_imgs.appendleft(b'')
        self.temp_imgs.clear()

    def seq_cancel(self):
        self.temp_imgs.clear()

    def set_config(self, date: str, time: str):
        with self.__data_lock:
            self.dateStamp = date
            self.timeStamp = time

    # -------- Job queue (NEW) --------

    def submit_job(self, payload, summary=None, job_id=None):
        """Append a new job to the queue. `payload` is the MATLAB
        getByteStreamFromArray(...) blob the runner will decode with
        getArrayFromByteStream. `summary` is an optional dict produced by
        ybScanSummary (axes, set_params, default_params, num_per_group,
        num_images, etc.) that the GUI uses to show scan details.

        `job_id` (pyctrl id-reuse): when given, the job takes that EXACT id
        instead of minting a fresh one, and the id counter is advanced past it so
        no future job collides. The pyctrl run loop passes the originating
        descriptor's id here so a scan carries a SINGLE id -- the one
        submit_scan_descriptor returned (and the .py scan script printed) --
        instead of a descriptor id plus a separate job id. link_descriptor_to_job
        then drops the now-redundant descriptor row (its same-id branch). Left
        None on the MATLAB ".m run-button" submit_job path, which has no
        descriptor -> unchanged fresh-id behavior. DIVERGENCE from
        matlab_new/YbExptCtrl/ExptServer.py (which always mints a fresh id and
        archives a distinct-id descriptor): pyctrl is both producer and consumer
        with an identity payload, so the second id was vestigial."""
        with self.__queue_lock:
            if job_id is None:
                jid = self.__next_job_id
                self.__next_job_id += 1
            else:
                jid = int(job_id)
                # Keep the counter strictly ahead of any reused id so a later
                # freshly-minted job can never collide with it.
                if jid >= self.__next_job_id:
                    self.__next_job_id = jid + 1
            entry = {
                'id': jid,
                'kind': 'job',                  # NEW: explicit discriminator
                'seqName': '',                  # set by runner via set_seq_name / below sniff
                'payload_size': len(payload),
                'payload': bytes(payload),
                'state': 'queued',
                'enqueued_ts': time.time(),
                'start_ts': None,
                'finish_ts': None,
                'status': None,
                'summary': summary if isinstance(summary, dict) else None,
            }
            # If the summary carries a scan filename, use it as a fallback
            # seqName display. The runner overwrites this with the real
            # seqName via set_seq_name() once it decodes the payload.
            if isinstance(summary, dict):
                sf = summary.get('scan_filename') or ''
                if sf:
                    entry['seqName'] = str(sf)
            if not entry['seqName']:
                # best-effort sniff from the payload head
                head = bytes(payload)[:4096]
                for known in (b'Seq', b'Scan'):
                    idx = head.rfind(known)
                    if idx > 0:
                        start = idx
                        while start > 0 and 32 <= head[start - 1] < 127 and head[start - 1] not in (0, 1, 2):
                            start -= 1
                        end = idx + len(known)
                        while end < len(head) and 32 <= head[end] < 127:
                            end += 1
                        cand = head[start:end].decode('latin-1', errors='ignore').strip()
                        if 3 <= len(cand) <= 64 and cand.replace('_', '').isalnum():
                            entry['seqName'] = cand
                            break
            self.__queue.append(entry)
            self.__save_queue_locked()
            return jid

    def queue_list(self) -> dict:
        with self.__queue_lock:
            queued = [self.__public_entry(e) for e in self.__queue if e['state'] == 'queued']
            running = next((self.__public_entry(e) for e in self.__queue if e['state'] == 'running'), None)
            history = [self.__public_entry(e) for e in self.__history]
            return {'queued': queued, 'running': running, 'history': history}

    def queue_remove(self, job_id: int) -> bool:
        """Remove a queued JOB (not a descriptor). Descriptors are
        cancelled via descriptor_remove. We restrict by kind so a job-id
        cancel doesn't accidentally hit a same-id descriptor (shouldn't
        happen -- shared id namespace -- but keep verbs distinct)."""
        with self.__queue_lock:
            for i, e in enumerate(self.__queue):
                if (e['id'] == job_id and e['state'] == 'queued'
                        and e.get('kind', 'job') == 'job'):
                    self.__queue.pop(i)
                    self.__save_queue_locked()
                    return True
            return False

    def queue_move(self, job_id: int, direction: str) -> bool:
        """direction = 'up' or 'down'. Moves a queued entry among queued
        SAME-KIND neighbors only -- jobs and descriptors don't compete
        for ordering (the dispatcher pops descriptors independently of
        the job runner)."""
        if isinstance(direction, (bytes, bytearray)):
            direction = direction.decode('ascii', errors='ignore')
        with self.__queue_lock:
            # Find the target entry first so we can match by id AND filter
            # the neighbor pool by the same kind.
            target_kind = None
            for e in self.__queue:
                if e['id'] == job_id:
                    target_kind = e.get('kind', 'job')
                    break
            if target_kind is None:
                return False
            queued_idx = [i for i, e in enumerate(self.__queue)
                          if e['state'] == 'queued'
                          and e.get('kind', 'job') == target_kind]
            pos = None
            for p, i in enumerate(queued_idx):
                if self.__queue[i]['id'] == job_id:
                    pos = p
                    break
            if pos is None:
                return False
            if direction == 'up' and pos > 0:
                i1, i2 = queued_idx[pos], queued_idx[pos - 1]
            elif direction == 'down' and pos < len(queued_idx) - 1:
                i1, i2 = queued_idx[pos], queued_idx[pos + 1]
            else:
                return False
            self.__queue[i1], self.__queue[i2] = self.__queue[i2], self.__queue[i1]
            self.__save_queue_locked()
            return True

    def pop_next_job(self):
        """Called by MATLAB. Returns the payload of the next queued JOB
        (not descriptor), marks it running, sets start_ts. Returns None
        if no queued jobs exist. Descriptors are popped separately via
        pop_next_descriptor."""
        with self.__queue_lock:
            for e in self.__queue:
                if (e['state'] == 'queued'
                        and e.get('kind', 'job') == 'job'):
                    e['state'] = 'running'
                    e['start_ts'] = time.time()
                    self.__save_queue_locked()
                    return {'id': e['id'], 'payload': e['payload'], 'seqName': e['seqName']}
            return None

    def finish_job(self, job_id: int, status: str):
        """Mark a running job as done/error, move to history."""
        with self.__queue_lock:
            for i, e in enumerate(self.__queue):
                if e['id'] == job_id:
                    e['state'] = 'done' if status == 'ok' else 'error'
                    e['finish_ts'] = time.time()
                    e['status'] = status
                    self.__history.insert(0, e)
                    del self.__history[HISTORY_CAP:]
                    self.__queue.pop(i)
                    self.__save_queue_locked()
                    return True
            return False

    def set_job_file_id(self, job_id: int, file_id: str):
        """Runner calls this after ybBuildScanJob so the queue shows the
        data folder identifier (e.g. '20260424_011913')."""
        with self.__queue_lock:
            for e in self.__queue:
                if e['id'] == job_id:
                    e['file_id'] = str(file_id)
                    return True
            for e in self.__history:
                if e['id'] == job_id:
                    e['file_id'] = str(file_id)
                    return True
            return False

    def set_seq_name(self, job_id: int, name: str):
        """Runner calls this after decoding the payload so the UI shows the
        true sequence name (not the heuristic sniff)."""
        with self.__queue_lock:
            for e in self.__queue:
                if e['id'] == job_id:
                    e['seqName'] = name
                    self.__save_queue_locked()
                    return True
            for e in self.__history:
                if e['id'] == job_id:
                    e['seqName'] = name
                    return True
            return False

    # -------- Descriptor queue (Phase 3) --------
    #
    # Descriptors live in the same self.__queue list as jobs, distinguished
    # by `kind` ('job' default, 'descriptor' for new entries). The
    # SequenceRunner loop pops descriptors between jobs and calls
    # dispatch_descriptor.m, which builds a fresh ScanGroup from the JSON
    # body, serializes via ybBuildScanPayload, and feeds it back through
    # this same submit_job() path -- so the resulting job appears in the
    # queue and is processed exactly like any editor-submitted job.
    #
    # State transitions:
    #   queued    -> building   (pop_next_descriptor)
    #   building  -> built      (link_descriptor_to_job, after success)
    #   building  -> error      (finish_descriptor with non-ok status)
    #   queued    -> removed    (descriptor_remove)

    def submit_scan_descriptor(self, descriptor_json, label=''):
        """Append a descriptor to the queue. `descriptor_json` is a JSON
        string conforming to yb_analysis/scans/descriptor.schema.json.
        Returns the assigned descriptor id (shares the namespace with
        job ids; the queue's id counter advances)."""
        if isinstance(descriptor_json, (bytes, bytearray)):
            descriptor_json = descriptor_json.decode('utf-8', errors='replace')
        descriptor_json = str(descriptor_json)
        with self.__queue_lock:
            did = self.__next_job_id
            self.__next_job_id += 1
            # Best-effort label / seqName extraction so the queue UI shows
            # a meaningful name without forcing the caller to set `label`.
            seq_name = ''
            parsed = None
            try:
                parsed = json.loads(descriptor_json)
                if isinstance(parsed, dict):
                    s = parsed.get('seq')
                    if isinstance(s, str):
                        seq_name = s
                    elif isinstance(s, dict):
                        seq_name = str(s.get('@', '')) or ''
                    if not label:
                        label = str(parsed.get('label', '')) or ''
            except Exception:
                pass
            # Build the ybScanSummary-shaped dict (axes / set_params / reps / scan_name) the
            # dashboard queue panel reads. Best-effort: a malformed descriptor leaves it None
            # and the queue UI degrades gracefully.
            summary = None
            if isinstance(parsed, dict):
                try:
                    from scan_summary import build_descriptor_summary
                    summary = build_descriptor_summary(parsed)
                except Exception:
                    summary = None
            entry = {
                'id': did,
                'kind': 'descriptor',
                'seqName': seq_name,
                'label': str(label) if label else '',
                'descriptor': descriptor_json,
                'state': 'queued',
                'enqueued_ts': time.time(),
                'start_ts': None,
                'finish_ts': None,
                'status': None,
                'built_job_id': None,
                'error_message': None,
                'summary': summary,
                'payload_size': 0,
            }
            self.__queue.append(entry)
            self.__save_queue_locked()
            return did

    def pop_next_descriptor(self):
        """Called by MATLAB (SequenceRunner) between jobs. Returns the
        next 'queued' descriptor and atomically marks it 'building', or
        None if no descriptor is queued. Returns dict with id +
        descriptor (JSON string) so the dispatcher can feed it to
        jsondecode and start building."""
        with self.__queue_lock:
            for e in self.__queue:
                if e.get('kind') == 'descriptor' and e['state'] == 'queued':
                    e['state'] = 'building'
                    e['start_ts'] = time.time()
                    self.__save_queue_locked()
                    return {
                        'id': e['id'],
                        'descriptor': e['descriptor'],
                        'label': e.get('label', ''),
                    }
            return None

    def link_descriptor_to_job(self, desc_id, job_id):
        """Called by the dispatcher after a successful submit_job, to
        stamp the built_job_id on the descriptor row and transition
        building -> built. Then the descriptor row is archived to
        history so it doesn't clutter the active queue.

        Returns True on success, False if the descriptor wasn't found
        in 'building' state."""
        desc_id = int(desc_id)
        job_id = int(job_id)
        with self.__queue_lock:
            for i, e in enumerate(self.__queue):
                if (e.get('kind') == 'descriptor'
                        and e['id'] == desc_id
                        and e['state'] == 'building'):
                    if desc_id == job_id:
                        # pyctrl id-reuse: the job took over this EXACT id
                        # (submit_job was called with job_id=desc_id). A separate
                        # 'built' descriptor row would just duplicate the id, so
                        # drop the descriptor entirely -- the job of the same id
                        # is the scan's single record. (DIVERGENCE from
                        # matlab_new, which archives a distinct-id descriptor;
                        # only reachable when the caller reuses the id, i.e. the
                        # pyctrl run loop -- never the MATLAB path.)
                        self.__queue.pop(i)
                        self.__save_queue_locked()
                        return True
                    e['state'] = 'built'
                    e['built_job_id'] = job_id
                    e['finish_ts'] = time.time()
                    e['status'] = 'ok'
                    # Archive to history -- the descriptor's job has
                    # taken over visibility via its own queue row.
                    self.__history.insert(0, e)
                    del self.__history[HISTORY_CAP:]
                    self.__queue.pop(i)
                    self.__save_queue_locked()
                    return True
            return False

    def finish_descriptor(self, desc_id, status, error_message=None):
        """Called by the dispatcher (or SequenceRunner outer catch) when
        descriptor dispatch FAILS. Transitions building -> error and
        archives to history."""
        desc_id = int(desc_id)
        if isinstance(status, (bytes, bytearray)):
            status = status.decode('ascii', errors='replace')
        with self.__queue_lock:
            for i, e in enumerate(self.__queue):
                if e.get('kind') == 'descriptor' and e['id'] == desc_id:
                    e['state'] = 'error' if status != 'ok' else 'built'
                    e['finish_ts'] = time.time()
                    e['status'] = str(status) if status else 'error'
                    if error_message is not None:
                        e['error_message'] = str(error_message)
                    self.__history.insert(0, e)
                    del self.__history[HISTORY_CAP:]
                    self.__queue.pop(i)
                    self.__save_queue_locked()
                    return True
            return False

    def descriptor_remove(self, desc_id):
        """Cancel a queued descriptor. Only valid while state=='queued'.
        Once a descriptor enters 'building' the dispatcher owns it."""
        desc_id = int(desc_id)
        with self.__queue_lock:
            for i, e in enumerate(self.__queue):
                if (e.get('kind') == 'descriptor'
                        and e['id'] == desc_id
                        and e['state'] == 'queued'):
                    self.__queue.pop(i)
                    self.__save_queue_locked()
                    return True
            return False

    # -------- Camera (called by MATLAB SequenceRunner loop) --------

    def get_camera_cmd(self):
        """Pop the pending camera command (dict or None)."""
        with self.__camera_lock:
            cmd = self._camera_pending
            self._camera_pending = None
            return cmd

    def set_camera_result(self, connected, roi, error='', exposure_time=None):
        """SequenceRunner reports result of a camera command.

        `exposure_time` is optional; when None the last known value is kept.
        MATLAB passes a scalar (seconds); pass py.None from MATLAB to leave
        unchanged."""
        with self.__camera_lock:
            self._camera['connected'] = bool(connected)
            self._camera['roi'] = list(roi) if roi else [0, 0, 0, 0]
            self._camera['error'] = str(error) if error else ''
            if exposure_time is not None:
                try:
                    self._camera['exposure_time'] = float(exposure_time)
                except (TypeError, ValueError):
                    pass

    def set_camera_status(self, status):
        """Merge a full camera-status dict into ``_camera`` (pyctrl run loop).

        Complements :meth:`set_camera_result` (which the MATLAB SequenceRunner calls with the
        connect/roi/error/exposure quartet): the pyctrl runner additionally pushes the extended
        Orca telemetry (trigger / cooler / cooler_status / temperature) so the monitor + web
        Camera card show a live, truthful state. Only the keys present in ``status`` are
        updated; unknown keys are ignored. ``error`` is cleared when not supplied (a healthy
        status push implies no error). Best-effort: a non-dict argument is ignored."""
        if not isinstance(status, dict):
            return
        allowed = ('connected', 'roi', 'exposure_time', 'error',
                   'trigger', 'cooler', 'cooler_status', 'temperature')
        with self.__camera_lock:
            for key in allowed:
                if key in status:
                    self._camera[key] = status[key]
            if 'connected' in status and 'error' not in status:
                self._camera['error'] = ''

    def dummy_enabled(self) -> bool:
        """Backward-compat wrapper: True iff mode is not 'off'."""
        with self.__dummy_lock:
            return self.__dummy_mode != 'off'

    def dummy_mode(self) -> str:
        """MATLAB polls this every idle iteration. Returns one of
        'off' | 'default' | 'last'."""
        with self.__dummy_lock:
            return self.__dummy_mode

    def last_seq_available(self) -> bool:
        """MATLAB-callable convenience: True iff the runner has notified us
        of a cached last seq."""
        with self.__dummy_lock:
            return bool(self.__last_seq_meta.get('available'))

    def set_last_seq_meta_direct(self, name: str, file_id: str):
        """MATLAB calls this directly (not via ZMQ) right after capturing
        the first iteration of a successful job."""
        with self.__dummy_lock:
            self.__last_seq_meta = {
                'available': True,
                'name': str(name) if name else '',
                'file_id': str(file_id) if file_id else '',
                'captured_at': time.time(),
            }

    def set_last_fallback_direct(self, active: bool):
        with self.__dummy_lock:
            self.__last_fallback_active = bool(active)

    def clear_pending_imgs(self):
        """Drop any unread image batches sitting in the deques. Used by
        the dummy/replay path to bound memory when no client is draining."""
        with self.__data_lock:
            with self.__expt_lock:
                self.expt_imgs.clear()
                self.imgs.clear()
                self.nseq_imgs = 0
        self.temp_imgs.clear()

    def __parse_camera_payload(self, payload_str):
        """camera_init accepts either a legacy JSON list (just ROI) or a
        dict {roi, exposure_time}. Returns (roi, exposure_or_None)."""
        if not payload_str:
            return [0, 0, 4096, 2304], None
        try:
            data = json.loads(payload_str)
        except Exception:
            return [0, 0, 4096, 2304], None
        if isinstance(data, list):
            return list(data), None
        if isinstance(data, dict):
            roi = data.get('roi') or [0, 0, 4096, 2304]
            exp = data.get('exposure_time')
            if exp is not None:
                try:
                    exp = float(exp)
                except (TypeError, ValueError):
                    exp = None
                else:
                    # Drop non-finite / non-positive — MATLAB treats None as
                    # "fall back to OrcaInit default", which is the safest
                    # behavior for bad startup config.
                    if not (0 < exp < float('inf')):
                        exp = None
            return list(roi), exp
        return [0, 0, 4096, 2304], None

    def __public_entry(self, e):
        return {
            'id': e['id'],
            'kind': e.get('kind', 'job'),     # NEW: 'job' (default) or 'descriptor'
            'seqName': e.get('seqName', ''),
            'state': e['state'],
            'enqueued_ts': e.get('enqueued_ts'),
            'start_ts': e.get('start_ts'),
            'finish_ts': e.get('finish_ts'),
            'status': e.get('status'),
            'payload_size': e.get('payload_size', 0),
            'summary': e.get('summary'),
            'file_id': e.get('file_id', ''),
            # Descriptor-only fields (omitted for jobs; consumers tolerate
            # missing keys).
            'descriptor': e.get('descriptor'),
            'built_job_id': e.get('built_job_id'),
            'label': e.get('label', ''),
            'error_message': e.get('error_message'),
        }

    def __save_queue_locked(self):
        """Caller must hold __queue_lock. Writes queue + history (payloads
        base64-encoded) atomically to JSON."""
        import base64
        import datetime
        try:
            _ensure_dir(QUEUE_PATH)
            serializable = {
                'date': datetime.date.today().isoformat(),
                'next_job_id': self.__next_job_id,
                'queue': [self.__entry_to_json(e) for e in self.__queue],
                'history': [self.__entry_to_json(e, with_payload=False) for e in self.__history],
            }
            tmp = QUEUE_PATH + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(serializable, f)
            os.replace(tmp, QUEUE_PATH)
        except Exception as ex:
            print(f"[ExptServer] warning: queue persist failed: {ex}")

    def __entry_to_json(self, e, with_payload=True):
        import base64
        d = dict(e)
        if with_payload and 'payload' in d and d['payload'] is not None:
            d['payload'] = base64.b64encode(d['payload']).decode('ascii')
        else:
            d.pop('payload', None)
        return d

    def __load_queue(self):
        """Called from __init__ under __queue_lock. Rehydrates queue from
        disk; any entry stuck in 'running' is demoted to 'queued'.
        If the saved date differs from today, reset the ID counter and
        clear history (queued jobs are kept)."""
        import base64
        import datetime
        if not os.path.exists(QUEUE_PATH):
            return
        try:
            with open(QUEUE_PATH, 'r') as f:
                data = json.load(f)
        except Exception as ex:
            print(f"[ExptServer] warning: could not load {QUEUE_PATH}: {ex}")
            return

        saved_date = data.get('date', '')
        today = datetime.date.today().isoformat()
        if saved_date != today:
            # New day — reset counter, discard history, keep queued jobs
            print(f"[ExptServer] new day ({saved_date} -> {today}): "
                  f"resetting job IDs, clearing history")
            self.__next_job_id = 1
            self.__history = []
            # Still reload any queued jobs (unlikely overnight but be safe).
            # Descriptors lose their built_job_id linkage across days (the
            # job IDs are renumbered), which is acceptable — by definition
            # they haven't been dispatched yet so no link exists.
            for raw in data.get('queue', []):
                e = dict(raw)
                kind = e.get('kind', 'job')
                if kind not in ('job', 'descriptor'):
                    continue   # skip unknown kinds on downgrade
                if 'payload' in e and isinstance(e['payload'], str):
                    try:
                        e['payload'] = base64.b64decode(e['payload'])
                    except Exception:
                        e['payload'] = b''
                e['state'] = 'queued'
                e['start_ts'] = None
                e['built_job_id'] = None   # any prior link is stale
                e['id'] = self.__next_job_id
                self.__next_job_id += 1
                self.__queue.append(e)
            self.__save_queue_locked()
            return

        self.__next_job_id = max(int(data.get('next_job_id', 1)), 1)
        demoted = 0
        demoted_desc = 0
        skipped_unknown = 0
        for raw in data.get('queue', []):
            e = dict(raw)
            kind = e.get('kind', 'job')
            # Downgrade safety: unknown future kinds are warned + skipped
            # rather than crashing the runner (Phase 3 plan: pre-Phase-3
            # code must skip unknown 'kind' rows).
            if kind not in ('job', 'descriptor'):
                skipped_unknown += 1
                continue
            if 'payload' in e and isinstance(e['payload'], str):
                try:
                    e['payload'] = base64.b64decode(e['payload'])
                except Exception:
                    e['payload'] = b''
            state = e.get('state')
            if kind == 'job' and state == 'running':
                e['state'] = 'queued'
                e['start_ts'] = None
                demoted += 1
            elif kind == 'descriptor' and state == 'building':
                # Mid-dispatch crash recovery: a 'building' descriptor
                # never made it to a job, so re-queue it for another
                # dispatch attempt.
                e['state'] = 'queued'
                e['start_ts'] = None
                demoted_desc += 1
            self.__queue.append(e)
        for raw in data.get('history', []):
            h = dict(raw)
            if h.get('kind') not in (None, 'job', 'descriptor'):
                skipped_unknown += 1
                continue
            self.__history.append(h)
        if demoted:
            print(f"[ExptServer] demoted {demoted} in-flight job(s) back to queued on reload")
        if demoted_desc:
            print(f"[ExptServer] demoted {demoted_desc} mid-dispatch descriptor(s) back to queued")
        if skipped_unknown:
            print(f"[ExptServer] WARNING: skipped {skipped_unknown} queue row(s) with unknown 'kind' "
                  f"(future schema? safe to ignore on downgrade)")
