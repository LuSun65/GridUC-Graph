"""Label all numbered samples by default; use --sample-id to select one."""

import argparse
from datetime import datetime
import logging
import re
import sys
import time
from pathlib import Path

from lib.label_pricing import label_sample

# dir example: /home/npg0/Sunlu/GridUC-Graph/data/case5/samples/tcuc
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--sample-id", type=int, help="Process only this sample ID; defaults to all samples")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate valid labels too")
    args = parser.parse_args()
    args.sample_dir = args.sample_dir.resolve()
    uc_type = args.sample_dir.name
    if uc_type not in ("tcuc", "topo", "scuc"):
        parser.error("Sample directory must end in tcuc, topo, or scuc (e.g. case5/samples/tcuc)")
    case_name = next(
        (part for part in reversed(args.sample_dir.parent.parts)
         if re.fullmatch(r"case[0-9]+", part)), None,
    )
    if case_name is None:
        parser.error("Sample path must contain a case directory such as case5 or case118")
    samples = sorted(
        (path for path in args.sample_dir.glob("*.pkl") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    if args.sample_id is not None:
        samples = [path for path in samples if int(path.stem) == args.sample_id]
    if not samples:
        parser.error("No matching numbered sample files found")
    log_dir = Path(__file__).resolve().parent / "runs" / "lmp_labels" / case_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{uc_type}_{datetime.now():%Y%m%d_%H%M%S}.log"
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
    try:
        logger.info("Log: %s", log_path)
        logger.info("START sample_dir=%s uc_type=%s overwrite=%s samples=%s",
                    args.sample_dir.resolve(), uc_type, args.overwrite,
                    [path.stem for path in samples])
        for index, path in enumerate(samples, 1):
            sample_started = time.monotonic()
            logger.info("[%d/%d] PROCESSING sample %s: %s", index, len(samples), path.stem, path)
            try:
                status = label_sample(path, uc_type, overwrite=args.overwrite)
            except Exception as error:
                reason = f"{type(error).__name__}: {error}"
                failed.append((path, reason))
                logger.error("FAILED sample %s (%.3fs): %s", path.stem,
                             time.monotonic() - sample_started, reason)
            else:
                saved += status == "saved"
                skipped += status == "skipped"
                logger.info("%s sample %s (%.3fs)", status.upper(), path.stem,
                            time.monotonic() - sample_started)
        logger.info("FINISHED total=%d saved=%d skipped=%d failed=%d elapsed=%.3fs",
                    len(samples), saved, skipped, len(failed), time.monotonic() - started)
        logger.info("Failed sample IDs: %s", ", ".join(path.stem for path, _ in failed) or "none")
        for path, reason in failed:
            logger.error("Failure summary: %s | %s", path, reason)
        return 1 if failed else 0
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()



if __name__ == "__main__":
    raise SystemExit(main())
