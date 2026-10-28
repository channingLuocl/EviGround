"""Download the four v10 ScanNet training files per scene onto an external disk."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
from pathlib import Path
import re
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = ('.txt', '.aggregation.json', '_vh_clean_2.0.010000.segs.json', '_vh_clean_2.ply')
BASE_URL = 'https://kaldir.vc.in.tum.de/scannet/v2/scans'


def download(scene, suffix, output, workers):
    destination = output / scene / (scene + suffix)
    if destination.is_file() and destination.stat().st_size:
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + '.part')
    for attempt in range(2):
        free = shutil.disk_usage(output).free - 5 * 1024**3
        if free <= 0:
            raise RuntimeError('磁盘可用空间不足 5 GiB，已停止下载')
        previous = part.stat().st_size if part.exists() else 0
        result = subprocess.run([
            'curl', '--location', '--fail', '--silent', '--show-error',
            '--connect-timeout', '20', '--max-time', '600',
            '--speed-limit', '1024', '--speed-time', '60',
            '--retry', '3', '--retry-delay', '2',
            '--continue-at', '-', '--max-filesize', str(free // workers),
            '--output', str(part), '--write-out', '%{http_code}',
            f'{BASE_URL}/{scene}/{scene + suffix}',
        ], capture_output=True, text=True)
        # A server without Range support (or a complete .part) needs a fresh request.
        if previous and attempt == 0 and (result.returncode == 33 or result.stdout == '416'):
            part.unlink(missing_ok=True)
            continue
        if result.returncode or result.stdout not in ('200', '206'):
            raise RuntimeError(f'{destination.name}: curl {result.returncode}, '
                               f'HTTP {result.stdout}; {result.stderr.strip()}')
        if not part.is_file() or not part.stat().st_size:
            raise RuntimeError(f'下载结果为空：{destination.name}')
        size = part.stat().st_size
        part.replace(destination)
        return max(0, size - previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/Volumes/KESU/ScanNet_v10/raw'))
    parser.add_argument('--scene-list', type=Path, default=ROOT / 'data/scannet/scannetv2_train.txt')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, help='Only download this many scenes for a trial')
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error('workers 和 limit 必须大于零')
    output = args.output.expanduser().absolute()
    if len(output.parts) < 4 or output.parts[1] != 'Volumes':
        parser.error('输出目录必须位于 /Volumes 下的外置硬盘')
    volume = Path(*output.parts[:3])
    if not volume.is_mount():
        parser.error(f'外置硬盘未挂载：{volume}')
    scenes = sorted(set(args.scene_list.read_text().split()), reverse=True)
    if not scenes or any(not re.fullmatch(r'scene\d{4}_\d{2}', s) for s in scenes):
        parser.error('场景清单必须包含有效的 ScanNet scene ID')
    scenes = scenes[:args.limit]
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.download.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('此目录已有下载任务运行，请勿重复启动')

        started, transferred, failed = time.monotonic(), 0, []
        print(f'输出：{output}\n场景数：{len(scenes)}，并发：{args.workers}；从后往前下载', flush=True)

        def acquire(scene):
            return sum(download(scene, suffix, output, args.workers) for suffix in SUFFIXES)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(acquire, scene): scene for scene in scenes}
            for index, future in enumerate(as_completed(jobs), 1):
                scene = jobs[future]
                try:
                    transferred += future.result()
                    seconds = time.monotonic() - started
                    print(f'[{index}/{len(scenes)}] 已就绪：{scene} | '
                          f'本轮新下载 {transferred / 1e6:.1f} MB | '
                          f'平均 {transferred / max(seconds, .001) / 1e6:.2f} MB/s', flush=True)
                except Exception as exc:
                    failed.append(scene)
                    print(f'[{index}/{len(scenes)}] 失败：{scene} | {exc}', flush=True)
        if failed:
            print(f'失败场景：{", ".join(failed)}。重新执行同一命令可补下。', flush=True)
            return 1
        print('全部下载完成。原始文件可上传至服务器 raw/；本脚本不执行转换或训练。', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
