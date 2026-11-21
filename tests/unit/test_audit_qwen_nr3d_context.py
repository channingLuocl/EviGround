"""Check bounded restoration of finalized Qwen audit artifacts."""
import io
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from audit_qwen_nr3d_context import iter_episodes


def write_archive(path, member, data):
    with tarfile.open(path, 'w:xz') as archive:
        info = tarfile.TarInfo(member)
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))


def test_audit_reads_live_and_archived_episodes_one_shard_at_a_time(tmp_path):
    output = tmp_path/'output'
    live = output/'q00003'/'episode'
    live.mkdir(parents=True)
    archives = []
    for name in ('q00001', 'q00002'):
        sample = output/name
        sample.mkdir()
        (sample/'evaluation.json').write_text('{}')
        path = tmp_path/f'{name}.tar.xz'
        write_archive(path, f'{name}/episode/receipt.json', name.encode())
        archives.append(path)

    episodes = iter_episodes(output, archives)
    assert next(episodes) == (live.parent, live)
    sample, first = next(episodes)
    assert sample.name == 'q00001'
    assert (first/'receipt.json').read_text() == sample.name
    sample, second = next(episodes)
    assert not first.exists()
    assert sample.name == 'q00002'
    assert (second/'receipt.json').read_text() == sample.name
    assert list(episodes) == []
    assert not second.exists()


@pytest.mark.parametrize('member', [
    'q00001/episode/../../unexpected.json',
    'q00002/episode/receipt.json',
    'q00001/evaluation.json',
])
def test_audit_rejects_archive_members_outside_finalized_episodes(tmp_path, member):
    output = tmp_path/'output'
    sample = output/'q00001'
    sample.mkdir(parents=True)
    (sample/'evaluation.json').write_text('{}')
    path = tmp_path/'episodes.tar.xz'
    write_archive(path, member, b'{}')

    with pytest.raises(ValueError, match='invalid or duplicate archived episode member'):
        list(iter_episodes(output, [path]))
