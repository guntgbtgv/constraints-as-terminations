from pathlib import Path
import csv
import time

import numpy as np
import mujoco
import mujoco.viewer

SCRIPT_DIR = Path(__file__).resolve().parent
# XML_PATH = SCRIPT_DIR / "urdf" / "scene.xml"
XML_PATH = SCRIPT_DIR / "urdf/go2_model" / "scene.xml"
# CSV_PATH = SCRIPT_DIR / "logs/clean_rl/articulate-trunk-quadruped_flat/2026-07-10_13-42-33/trajectory_log.csv"
CSV_PATH = SCRIPT_DIR / "logs/skrl/go2_cat/2026-08-20_14-58-48_ppo_torch/trajectory_log.csv"

def read_csv_trajectory(csv_path: Path):
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            raise RuntimeError(f"No data rows found in {csv_path}")
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise RuntimeError(f"No header found in {csv_path}")
    return rows, fieldnames

def main():
    rows, fieldnames = read_csv_trajectory(CSV_PATH)

    joint_pos_cols = [c for c in fieldnames if c.startswith("joint_pos_")]
    if not joint_pos_cols:
        raise RuntimeError("No joint_pos_* columns found in the CSV.")

    logged_joint_names = [c[len("joint_pos_"):] for c in joint_pos_cols]
    logged_joint_index = {name: i for i, name in enumerate(logged_joint_names)}

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # MuJoCo joint names in model order
    mj_joint_names = []
    mj_joint_qposadr = {}
    mj_joint_type = {}

    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name is not None:
            mj_joint_names.append(name)
            mj_joint_qposadr[name] = model.jnt_qposadr[jid]
            mj_joint_type[name] = model.jnt_type[jid]

    print("Logged joint order:", logged_joint_names)
    print("MuJoCo joint order:", mj_joint_names)

    # Playback speed
    dt = 0.005
    if len(rows) >= 2 and "t" in rows[0] and "t" in rows[1]:
        try:
            dt_candidate = float(rows[1]["t"]) - float(rows[0]["t"])
            if dt_candidate > 0.0:
                dt = dt_candidate
        except Exception:
            pass

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for row in rows:
            step_start = time.time()

            # Base pose from CSV
            root_pos = np.array(
                [float(row["root_pos_x"]), float(row["root_pos_y"]), float(row["root_pos_z"])],
                dtype=np.float64,
            )
            root_quat_wxyz = np.array(
                [float(row["root_quat_w"]), float(row["root_quat_x"]), float(row["root_quat_y"]), float(row["root_quat_z"])],
                dtype=np.float64,
            )

            data.qpos[:] = 0.0
            data.qvel[:] = 0.0

            # freejoint: [x, y, z, w, x, y, z]
            data.qpos[:3] = root_pos
            data.qpos[3:7] = root_quat_wxyz

            # joint positions in logged order
            joint_pos_logged = np.array([float(row[col]) for col in joint_pos_cols], dtype=np.float64)
            logged_value_by_name = {name: joint_pos_logged[i] for i, name in enumerate(logged_joint_names)}

            # Fill all non-free joints by name
            for name in mj_joint_names:
                if mj_joint_type[name] == mujoco.mjtJoint.mjJNT_FREE:
                    continue
                qadr = mj_joint_qposadr[name]
                data.qpos[qadr] = logged_value_by_name[name]

            mujoco.mj_forward(model, data)
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
                viewer.cam.lookat[:] = data.qpos[:3]
                viewer.cam.elevation = 0
            viewer.sync()

            # Maintain real-time step duration
            time_until_next_step = 5*dt - (time.time() - step_start)
            if time_until_next_step > 0:
                # print(time_until_next_step)
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()