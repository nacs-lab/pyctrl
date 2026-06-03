"""Controlled single-shot LACScan validation -- NEEDS-HARDWARE (fires the experiment ONCE).

Isolates the scenario-3 capture chain BEFORE involving the monitor/dashboard: load config,
open + arm the Orca, run ONE shot of TweezerLoadingSeq (the seq LACScan uses) through the real
engine + FPGA, capture the 1 image via the run-loop capture path (make_engine_run), store it to
a LOCAL ExptServer, and verify get_imgs() returns exactly one image.

⚠ This DRIVES THE FULL EXPERIMENT: TweezerLoadingSeq runs Init -> BlueMOT -> SLM -> GreenMOT ->
LAC -> Imag399 -> Init (MOT load, SLM tweezers, LAC, 399 imaging), arms the NI DAQ (14 analog
channels), and drives every FPGA/NI channel to its expConfig default. Run only with MATLAB OFF,
the camera free, and a confirmed-safe hardware state.

Params mirror LACScan.m's active settings (BlueMOT.LoadingTime=0.4, GreenMOT.CoolDown.HoldTime
=0.1; runp NumImages=1, NumPerGroup=500, isInit=1). rep=1 -> a single shot -> a single image.
A small ROI is used so the validation payload is small; the real monitor run sets the imaging
ROI via camera_init.

Usage:  pyctrl/.venv-engine/Scripts/python tools/run_lacscan_once.py
"""

import os
import socket
import sys


def _bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../pyctrl
    for d in ("lib", "YbExptCtrl", "YbSeqs", "YbSteps"):
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)


def build_lac_scangroup():
    """The LACScan ScanGroup (single point, no sweep) -- mirrors LACScan.m's active params."""
    from scan_group import ScanGroup
    g = ScanGroup()
    g().BlueMOT.LoadingTime = 0.4
    g().GreenMOT.CoolDown.HoldTime = 0.1
    rp = g.runp()
    rp.NumPerGroup = 500.0
    rp.NumImages = 1.0
    rp.isInit = 1.0
    rp.Scramble = 0.0
    rp.isHC = 0.0
    rp.isGrid2 = 0.0
    return g


def _save_png(frame, path, title=""):
    """Save a viewable PNG (jet colormap + colorbar) of the captured frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(frame, cmap="jet")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(roi=(1000, 100, 2100, 2100), exposure=0.050004):
    # Defaults = the real imaging ROI + exposure from expConfig (consts.Orca.ROI /
    # ExposureTime), so the captured frame covers the tweezer array (atoms), not background.
    _bootstrap()
    import numpy as np
    import runner
    from seq_config import SeqConfig
    from orca_camera import OrcaCamera
    from control_channel import ControlChannel
    from ExptServer import ExptServer
    from TweezerLoadingSeq import TweezerLoadingSeq

    runner.load_configs(log=lambda m: print("[cfg]", m))

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    url = "tcp://127.0.0.1:%d" % s.getsockname()[1]
    s.close()
    server = ExptServer(url)
    cam = OrcaCamera(roi=list(roi), exposure=exposure)
    try:
        print("[cam] temp=%.1fC roi=%s exp=%.4fs" % (
            cam.get_temperature(), cam.current_roi(), cam.get_exposure()))
        run = runner.make_engine_run(server, cam, SeqConfig.get())
        g = build_lac_scangroup()
        control = ControlChannel(server)
        print("[run] firing ONE shot of TweezerLoadingSeq ...")
        res = run(TweezerLoadingSeq, g, control=control, rep=1)
        print("[run] result:", res)

        n_imgs = server.get_num_imgs()
        n_seq = server.get_seq_num()
        print("[verify] num_imgs(nseq_imgs)=%d  seq_num(nseq)=%d" % (n_imgs, n_seq))
        raw = server.get_imgs()
        arr = np.frombuffer(bytes(raw), dtype="<f8")
        nseqs = int(arr[0]) if arr.size else 0
        print("[verify] get_imgs: nseqs=%d  total_doubles=%d" % (nseqs, arr.size))
        if nseqs >= 1 and arr.size >= 6:
            scan_id, seq_id = arr[1], arr[2]
            s1, s2, s3 = int(arr[3]), int(arr[4]), int(arr[5])
            print("[verify] image shape=(%d,%d,%d) scan_id=%g seq_id=%g" % (s1, s2, s3, scan_id, seq_id))
            npix = s1 * s2 * s3
            if npix > 0 and arr.size >= 6 + npix:
                frame = arr[6:6 + npix].reshape(s1, s2, s3, order="F")[:, :, 0]
                here = os.path.dirname(os.path.abspath(__file__))
                np.save(os.path.join(here, "last_lacscan.npy"), frame)
                _save_png(frame, os.path.join(here, "last_lacscan.png"),
                          title="LACScan shot: max=%d mean=%.1f" % (int(frame.max()), frame.mean()))
                print("[verify] frame stats: min=%d max=%d mean=%.1f -> saved last_lacscan.png/.npy"
                      % (int(frame.min()), int(frame.max()), frame.mean()))
                print("[RESULT] PASS -- one image captured + served")
            else:
                print("[RESULT] FAIL -- empty image")
        else:
            print("[RESULT] FAIL -- no image reached the server")
    finally:
        try:
            cam.close()
        except Exception as e:
            print("[cleanup] camera close err:", e)
        try:
            server.stop_worker()
        except Exception:
            pass
    print("[done]")


if __name__ == "__main__":
    main()
