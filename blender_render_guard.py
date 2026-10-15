#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import shutil
import time
import math
import logging
import signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_cuda_devices():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True, text=True, check=True
        )
        devices = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        if devices:
            return devices
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return ['0'] # fallback to single device '0' if no nvidia-smi

def partition_frames(start, end, num_chunks):
    total_frames = end - start + 1
    if total_frames <= 0:
        return []

    chunks = []
    frames_per_chunk = math.ceil(total_frames / num_chunks)
    current_start = start
    for _ in range(num_chunks):
        if current_start > end:
            break
        current_end = min(current_start + frames_per_chunk - 1, end)
        chunks.append((current_start, current_end))
        current_start = current_end + 1
    return chunks

def verify_image(filepath):
    """Verifies that an image file exists, is non-empty, and has a valid PNG or EXR header."""
    if not os.path.exists(filepath):
        return False
    if os.path.getsize(filepath) == 0:
        return False

    try:
        with open(filepath, 'rb') as f:
            header = f.read(8)
            # PNG signature
            if header.startswith(b'\x89PNG\r\n\x1a\n'):
                return True
            # EXR signature (magic number)
            if header.startswith(b'\x76\x2f\x31\x01'):
                return True
    except IOError:
        return False

    return False

def verify_output_frames(output_dir):
    if not output_dir or not os.path.exists(output_dir):
        return True # nothing to verify

    valid_files = 0
    for root, _, files in os.walk(output_dir):
        for file in files:
            filepath = os.path.join(root, file)
            if filepath.lower().endswith(('.png', '.exr')):
                if not verify_image(filepath):
                    logger.error(f"Integrity check failed for {filepath}")
                    return False
                valid_files += 1

    if valid_files == 0:
        logger.warning(f"No valid PNG or EXR files found in {output_dir}")
        return False
    else:
        logger.info(f"Verified {valid_files} output files successfully.")

    return True

def shutdown_system(grace_period):
    logger.info(f"Shutting down system in {grace_period} seconds...")
    time.sleep(grace_period)
    # Check OS
    if sys.platform == "win32":
        subprocess.run(['shutdown', '/s', '/t', '0'], check=False)
    else:
        subprocess.run(['sudo', 'shutdown', '-h', 'now'], check=False)

def main():
    parser = argparse.ArgumentParser(description="Blender Headless Multi-GPU Render Watchdog with Credit Guard")
    parser.add_argument('--blend-file', required=True, help="Path to the .blend file")
    parser.add_argument('--start-frame', type=int, required=True, help="Start frame")
    parser.add_argument('--end-frame', type=int, required=True, help="End frame")
    parser.add_argument('--output-dir', help="Temporary output directory for rendering")
    parser.add_argument('--dest-dir', help="Final destination directory to copy rendered frames")
    parser.add_argument('--no-shutdown', action='store_true', help="Disable automatic system shutdown")
    parser.add_argument('--grace-period', type=int, default=15, help="Shutdown grace period in seconds (default: 15)")
    parser.add_argument('--render-crash-log', default='render_crash.log', help="Path to write crash log")

    args = parser.parse_args()

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    devices = get_cuda_devices()
    logger.info(f"Found CUDA devices: {devices}")

    chunks = partition_frames(args.start_frame, args.end_frame, len(devices))
    logger.info(f"Partitioned frames into chunks: {chunks}")

    if not chunks:
        logger.error("No frames to render.")
        sys.exit(1)

    processes = []

    for i, chunk in enumerate(chunks):
        device = devices[i]
        start, end = chunk

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = device

        cmd = ['blender', '-b', args.blend_file]
        if args.output_dir:
            cmd.extend(['-o', os.path.join(args.output_dir, 'frame_######')])

        cmd.extend(['-f', f"{start}..{end}"])

        logger.info(f"Starting worker on GPU {device} for frames {start}-{end}")

        # Redirect stderr to a temp file for this worker to prevent pipe buffer deadlock
        err_file_path = f"worker_gpu_{device}_{start}_{end}_stderr.log"
        err_file = open(err_file_path, 'w')

        p = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL, # We don't need stdout, discard it to prevent blocking
            stderr=err_file,
            text=True
        )
        processes.append((p, device, start, end, err_file, err_file_path))

    failed = False
    crash_log_written = False

    # Monitor processes
    while processes:
        for p_info in processes[:]:
            p, device, start, end, err_file, err_file_path = p_info
            ret = p.poll()
            if ret is not None:
                # Process finished
                processes.remove(p_info)
                err_file.close() # Close file handle

                if ret != 0:
                    logger.error(f"Worker on GPU {device} (frames {start}-{end}) crashed with exit code {ret}.")
                    if not crash_log_written:
                        try:
                            with open(err_file_path, 'r') as f:
                                stderr_content = f.read()
                            with open(args.render_crash_log, 'w') as f:
                                f.write(stderr_content)
                            logger.info(f"Crash log written to {args.render_crash_log}")
                            crash_log_written = True
                        except Exception as e:
                            logger.error(f"Failed to read/write crash log: {e}")
                    failed = True
                    break # Break out of inner loop to terminate remaining
                else:
                    logger.info(f"Worker on GPU {device} (frames {start}-{end}) completed successfully.")
                    # cleanup temp err log on success
                    try:
                        os.remove(err_file_path)
                    except OSError:
                        pass

        if failed:
            # Terminate remaining processes cleanly
            for p_info in processes:
                p, device, start, end, err_file, err_file_path = p_info
                logger.warning(f"Terminating worker on GPU {device} (frames {start}-{end})...")
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                err_file.close()
            break

        time.sleep(1)

    if failed:
        logger.error("Rendering failed due to worker crash.")
        sys.exit(1)

    # Verify integrity
    if args.output_dir:
        logger.info("Verifying output frame integrity...")
        if not verify_output_frames(args.output_dir):
            logger.error("Output frame verification failed.")
            sys.exit(1)

    # Copy to destination if specified
    if args.output_dir and args.dest_dir:
        logger.info(f"Copying rendered frames to {args.dest_dir}...")
        if not os.path.exists(args.dest_dir):
            os.makedirs(args.dest_dir)
        try:
            for item in os.listdir(args.output_dir):
                s = os.path.join(args.output_dir, item)
                d = os.path.join(args.dest_dir, item)
                if os.path.isfile(s):
                    shutil.copy2(s, d)
            logger.info("Copy successful.")
        except Exception as e:
            logger.error(f"Failed to copy files: {e}")
            sys.exit(1)

    # Auto-shutdown & Credit Guard
    if not args.no_shutdown:
        shutdown_system(args.grace_period)
    else:
        logger.info("Auto-shutdown bypassed via --no-shutdown flag.")

    logger.info("All tasks completed successfully.")

if __name__ == "__main__":
    main()
