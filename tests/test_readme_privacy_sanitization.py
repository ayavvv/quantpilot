import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README_EN = REPO_ROOT / 'README.md'
README_ZH = REPO_ROOT / 'README_CN.md'


PRIVATE_LAN_IP_RE = re.compile(r"192\.168\.(?!x\.x)\d{1,3}\.\d{1,3}")
PRIVATE_GITHUB_USER_RE = re.compile(r"github\.com/(?!your-username)[A-Za-z0-9_.-]+/quantpilot\.git")


def test_readmes_do_not_contain_specific_private_examples():
    en = README_EN.read_text()
    zh = README_ZH.read_text()
    combined = en + "\n" + zh

    assert PRIVATE_LAN_IP_RE.search(combined) is None
    assert PRIVATE_GITHUB_USER_RE.search(combined) is None
    assert 'SSH_KEY=~/.ssh/' not in combined


def test_readmes_keep_generic_placeholders():
    en = README_EN.read_text()
    zh = README_ZH.read_text()

    assert 'FUTU_HOST=<your-futu-host>' in en
    assert 'FUTU_HOST=<your-futu-host>' in zh
    assert 'SSH_KEY=/path/to/ssh_private_key' in en
    assert 'SSH_KEY=/path/to/ssh_private_key' in zh
    assert 'https://github.com/your-username/quantpilot.git' in zh
