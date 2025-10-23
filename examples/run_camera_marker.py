import time
import numpy as np

from realsense_toolbox.camera import Camera
from hand_eye_calibration.config import serial, camera_config
from hand_eye_calibration.utils import get_marker_detector, detect_marker_in_image, estimate_marker_pose

# --- intrinsics shim to match your reference style ---
def get_camera_intrinsics(cam):
    """
    Return (K, D) for the color stream, matching your reference's 'get_camera_intrinsics()'.
    """
    # try common attributes your RealSense wrapper might expose
    for k_attr, d_attr in [("color_K", "color_D"), ("K_color", "D_color"), ("K", "D")]:
        K = getattr(cam.rs_camera, k_attr, None)
        D = getattr(cam.rs_camera, d_attr, None)
        if K is not None and D is not None:
            return (np.array(K, dtype=np.float32).reshape(3, 3),
                    np.array(D, dtype=np.float32).reshape(-1, 1))

    intr = getattr(cam.rs_camera, "color_intrinsics", None)
    if intr is not None and hasattr(intr, "fx"):
        K = np.array([[intr.fx, 0, intr.ppx],
                      [0, intr.fy, intr.ppy],
                      [0,      0,       1]], dtype=np.float32)
        D = np.array(getattr(intr, "coeffs", [0, 0, 0, 0, 0])[:5], dtype=np.float32).reshape(-1, 1)
        return K, D

    get_intr = getattr(cam.rs_camera, "get_color_intrinsics", None)
    if callable(get_intr):
        out = get_intr()
        if isinstance(out, (list, tuple)) and len(out) >= 2:
            return (np.array(out[0], dtype=np.float32).reshape(3, 3),
                    np.array(out[1], dtype=np.float32).reshape(-1, 1))
        if isinstance(out, dict):
            K = out.get("K") or out.get("color_K")
            D = out.get("D") or out.get("color_D")
            if K is not None and D is not None:
                return (np.array(K, dtype=np.float32).reshape(3, 3),
                        np.array(D, dtype=np.float32).reshape(-1, 1))

    return None, None


def make_overlays_from_detection(
    corners_list,
    ids,
    K=None,
    D=None,
    marker_length_m=0.05,
    draw_pose_text=True,
    color_box=(0, 255, 0),
):
    """
    Convert your detect_marker_in_image(...) outputs into overlay dicts for draw_overlays().
    Returns (overlays, (rvec, tvec)) for the FIRST detected marker (or (None, None)).
    """
    overlays = []

    if ids is None or len(ids) == 0:
        return overlays, (None, None)

    first_pose = (None, None)

    for i, marker_id in enumerate(ids.flatten()):
        # corners_list[i] may be (1,4,2) or (4,1,2). Normalize to (4,2) float32
        c = corners_list[i]
        c = np.asarray(c, dtype=np.float32)
        if c.shape == (1, 4, 2):
            c = c.reshape(4, 2)
        elif c.shape == (4, 1, 2):
            c = c.reshape(4, 2)
        elif c.shape != (4, 2):
            c = c.reshape(4, 2)  # last resort

        # four points, in image pixel coords
        pts = [(int(c[j, 0]), int(c[j, 1])) for j in range(4)]

        # bounding box
        overlays.append({
            "type": "box",
            "points": pts,
            "color": color_box,
            "thickness": 3
        })

        # optional pose (use your estimate_marker_pose signature)
        if K is not None and D is not None and first_pose == (None, None):
            try:
                rvec, tvec = estimate_marker_pose(
                    corners=c, marker_length=marker_length_m, camera_matrix=K, dist_coeffs=D
                )
                first_pose = (rvec, tvec)
                if draw_pose_text and rvec is not None and tvec is not None:
                    overlays.append({
                        "type": "text",
                        "text": f"r=[{rvec[0,0]:+.2f},{rvec[1,0]:+.2f},{rvec[2,0]:+.2f}]  "
                                f"t=[{tvec[0]:+.3f},{tvec[1]:+.3f},{tvec[2]:+.3f}] m",
                        "position": [20, 30],
                        "color": (255, 255, 255)
                    })
            except Exception as ex:
                overlays.append({
                    "type": "text",
                    "text": f"PnP fail: {ex}",
                    "position": [20, 80],
                    "color": (0, 0, 255)
                })

    return overlays, first_pose


def main():
    # ===== YOUR CHANGES =====
    MARKER_LENGTH_M = 0.05      # set to your printed marker side length (meters)
    DRAW_POSE_TEXT = True

    detector = get_marker_detector()  # your helper
    prev_overlays = []                # ALWAYS pass a list (possibly empty), not None
    K = None
    D = None
    # ========================

    camera = None
    try:
        camera = Camera(serial, camera_config)
        camera.launch()

        # grab intrinsics once; retry if None
        K, D = get_camera_intrinsics(camera)

        while True:
            # 1) show overlays from the *previous* frame
            camera.update(overlays=prev_overlays)

            # 2) read current frame, run your detector, build NEXT overlays
            color_image, depth_image, _, _ = camera.get_current_state()
            if color_image is not None:
                if K is None or D is None:
                    K, D = get_camera_intrinsics(camera)

                # your helper returns (annotated_img, corners_list, ids)
                # we'll ignore annotated_img and build overlays ourselves
                _, corners_list, ids = detect_marker_in_image(color_image, detector)

                next_overlays, (rvec, tvec) = make_overlays_from_detection(
                    corners_list, ids, K, D,
                    marker_length_m=MARKER_LENGTH_M, draw_pose_text=DRAW_POSE_TEXT
                )

                # store list (possibly empty), never None
                prev_overlays = next_overlays

            if not camera.is_alive:
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting gracefully.")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
    finally:
        if camera:
            camera.shutdown()
        print("Shutdown complete!")


if __name__ == "__main__":
    main()
