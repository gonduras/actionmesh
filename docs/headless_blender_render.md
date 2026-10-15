# Blender Headless Multi-GPU Render Watchdog with Credit Guard

`blender_render_guard.py` is a utility designed to maximize rendering throughput on multi-GPU cloud instances (such as iRender 4x RTX 4090 or RunPod) while safeguarding your rendering credits.

It handles multi-GPU workload distribution, monitors for crashes, validates output integrity, and automatically shuts down the instance upon successful completion to prevent idle billing.

## Features

* **Multi-GPU Splitting:** Automatically discovers all available CUDA devices using `nvidia-smi` and partitions frame ranges evenly across the available GPUs.
* **Isolated Execution:** Spawns headless `blender -b` instances per GPU utilizing the `CUDA_VISIBLE_DEVICES` environment variable for isolation.
* **Integrity Verification:** Checks that rendered output frames (PNG or EXR) are non-empty and contain valid headers before proceeding.
* **Credit Guard (Auto-Shutdown):** Upon successful verification, the script copies the frames to a destination directory and issues a system shutdown to stop instance billing. A configurable grace period allows time for the final files to sync.
* **Error Resilience:** In the event of a worker crash, the remaining render tasks are cleanly terminated and the stderr output is written to `render_crash.log`.

## Usage

```bash
python blender_render_guard.py \
    --blend-file <path_to_blend_file> \
    --start-frame <start> \
    --end-frame <end> \
    --output-dir <temp_render_dir> \
    --dest-dir <final_destination_dir>
```

### CLI Arguments

* `--blend-file` (Required): Path to the `.blend` file to be rendered.
* `--start-frame` (Required): The first frame to render.
* `--end-frame` (Required): The last frame to render.
* `--output-dir`: The temporary directory where Blender should render the frames.
* `--dest-dir`: The final destination directory where verified frames will be copied.
* `--no-shutdown`: Disables the automatic system shutdown upon successful completion.
* `--grace-period`: The shutdown grace period in seconds (default: 15).
* `--render-crash-log`: Path to write the stderr crash log if a worker fails (default: `render_crash.log`).

## Example

Render frames 1 through 72 on all available GPUs, verify the output, copy to `/mnt/storage/renders/`, and shutdown the machine in 30 seconds:

```bash
python blender_render_guard.py \
    --blend-file scene.blend \
    --start-frame 1 \
    --end-frame 72 \
    --output-dir /tmp/renders \
    --dest-dir /mnt/storage/renders/ \
    --grace-period 30
```

To run the same setup for testing purposes without triggering a shutdown:

```bash
python blender_render_guard.py \
    --blend-file scene.blend \
    --start-frame 1 \
    --end-frame 72 \
    --output-dir /tmp/renders \
    --dest-dir /mnt/storage/renders/ \
    --no-shutdown
```