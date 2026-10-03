# lesion-preservation

How fast can a brain MRI scan go before small lesions of each size are lost, and do PSNR and
SSIM warn in time? The Lesion Preservation Test runs any reconstruction method through a
fixed set of speed-ups and reports, lesion by lesion, whether each is still visible.

Status: under construction, nothing run.

## Data
fastMRI brain raw data and fastMRI+ lesion boxes. Neither is included here; users need
their own copy under the fastMRI data sharing agreement.

## Development
Python 3.13.

```
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e '.[dev]'
ruff check . && ruff format --check . && pytest -v
```

Tests that need the data files are skipped unless `LESION_PRESERVATION_DATA` points to them.

## License
MIT.
