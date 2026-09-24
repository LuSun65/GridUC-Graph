"""Label numbered samples in a directory or a single file via --sample-path."""

import argparse
from datetime import datetime
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path

from lib import lmp_sover
from lib.label_pricing import label_sample
from lib.toolkit import load_pkl


def _label_one(args):
    path, uc_type, overwrite = args
    started = time.monotonic()
    try:
        status = label_sample(path, uc_type, overwrite=overwrite)
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        return path, None, time.monotonic() - started, reason
    return path, status, time.monotonic() - started, None

# dir example: /home/npg0/Sunlu/GridUC-Graph/data/case5/samples/tcuc
# command example: python label_pricing.py --sample-path /home/npg0/Sunlu/GridUC-Graph-v2/samples/fail.pkl --overwrite
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample-dir", type=Path)
    source.add_argument("--sample-path", type=Path, help="Single .pkl file; infer case and UC type from its contents")
    parser.add_argument("--sample-id", type=int, help="Process only this sample ID; defaults to all samples")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate valid labels too")
    args = parser.parse_args()
    if args.sample_path is not None:
        if args.sample_id is not None:
            parser.error("--sample-id can only be used with --sample-dir")
        args.sample_path = args.sample_path.resolve()
        data = load_pkl(args.sample_path)
        case_name = data.case_name
        if data.cc_monitor:
            uc_type = "scuc"
        elif data.maintenance_line is not None:
            uc_type = "topo"
        else:
            uc_type = "tcuc"
        samples = [args.sample_path]
        args.sample_dir = args.sample_path.parent
    else:
        args.sample_dir = args.sample_dir.resolve()
        uc_type = args.sample_dir.name
        if uc_type not in ("tcuc", "topo", "scuc"):
            parser.error("Sample directory must end in tcuc, topo, or scuc (e.g. case5/samples/tcuc)")
        # Standard layout: <case>/samples/<uc_type>.
        case_name = args.sample_dir.parent.parent.name
        samples = sorted(
            (path for path in args.sample_dir.glob("*.pkl") if path.stem.isdigit()),
            key=lambda path: int(path.stem),
        )
        if args.sample_id is not None:
            samples = [path for path in samples if int(path.stem) == args.sample_id]
        if not samples:
            parser.error("No matching numbered sample files found")
    # Labels are saved in place, so the sample directory identifies both versions.
    versions = {"GridUC-Graph": "v1", "GridUC-Graph-v2": "v2"}
    version = next(
        (versions[part] for part in reversed(args.sample_dir.parts) if part in versions),
        None,
    )
    if version is None:
        parser.error("Sample path must be inside GridUC-Graph (v1) or GridUC-Graph-v2 (v2)")
    log_dir = Path(__file__).resolve().parent / "log" / case_name / "lmp_labels"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{uc_type}_{version}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = logging.getLogger("lmp_labels")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    failed = []
    saved = skipped = 0
    started = time.monotonic()
    workers = min(os.cpu_count() or 1, len(samples))
    try:
        logger.info("Log: %s", log_path)
        logger.info("Data read directory: %s", args.sample_dir)
        logger.info("Data save directory: %s (in-place update)", args.sample_dir)
        logger.info("PRICING_FEASIBILITY_TOL=%g PRICING_OPTIMALITY_TOL=%g",
                    lmp_sover.PRICING_FEASIBILITY_TOL,
                    lmp_sover.PRICING_OPTIMALITY_TOL)
        logger.info("START version=%s uc_type=%s overwrite=%s samples=%s",
                    version, uc_type, args.overwrite,
                    [path.stem for path in samples])
        logger.info("Using %d worker processes (detected CPUs=%d)",
                    workers, os.cpu_count() or 1)
        tasks = ((path, uc_type, args.overwrite) for path in samples)
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            for index, (path, status, elapsed, error) in enumerate(
                pool.imap_unordered(_label_one, tasks), 1
            ):
                logger.info("[%d/%d] FINISHED sample %s",
                            index, len(samples), path.stem)
                if error is not None:
                    reason = error
                    failed.append(path)
                    logger.error("FAILED sample %s (%.3fs): %s", path.stem,
                                 elapsed, reason)
                else:
                    saved += status == "saved"
                    skipped += status == "skipped"
                    logger.info("%s sample %s (%.3fs)", status.upper(), path.stem,
                                elapsed)
        logger.info("FINISHED total=%d saved=%d skipped=%d failed=%d elapsed=%.3fs",
                    len(samples), saved, skipped, len(failed), time.monotonic() - started)
        logger.info("Failed sample IDs: %s", ", ".join(path.stem for path in failed) or "none")
        return 1 if failed else 0
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()



if __name__ == "__main__":
    raise SystemExit(main())
