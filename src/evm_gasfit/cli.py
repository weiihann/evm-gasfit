"""Command-line entry point: ``evm-gasfit run …``."""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

from evm_gasfit.adapter.eest import prepare_eest
from evm_gasfit.adapter.zkevm import prepare_zkevm
from evm_gasfit.api import GasFit
from evm_gasfit.errors import ConfigError, ModelingError

_log = logging.getLogger("evm_gasfit")


def _install_logging() -> None:
    root = logging.getLogger("evm_gasfit")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
    root.setLevel(logging.WARNING)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evm-gasfit")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the full estimation pipeline")
    run_p.add_argument("--config", required=True, type=Path)
    run_p.add_argument("--runtimes", required=True, type=Path)
    run_p.add_argument("--opcounts", required=True, type=Path)
    run_p.add_argument(
        "--manifest",
        required=False,
        type=Path,
        default=None,
        help="Optional campaign manifest JSON; embedded into analysis_status.json",
    )
    run_p.add_argument("--out", required=True, type=Path)
    run_p.set_defaults(func=_run)

    compare_p = sub.add_parser(
        "compare-campaigns",
        help="Compare two completed analysis directories (manifest-aware)",
    )
    compare_p.add_argument("--baseline", required=True, type=Path)
    compare_p.add_argument("--candidate", required=True, type=Path)
    compare_p.add_argument("--out", required=True, type=Path)
    compare_p.set_defaults(func=_compare_campaigns)

    prepare_eest_p = sub.add_parser(
        "prepare-eest",
        help="Normalize EEST blockchain_tests fixtures into gasfit inputs",
    )
    prepare_eest_p.add_argument("--eest-fixtures", required=True, type=Path)
    prepare_eest_p.add_argument("--out", required=True, type=Path)
    prepare_eest_p.set_defaults(func=_prepare_eest)

    prepare_zkevm_p = sub.add_parser(
        "prepare-zkevm",
        help="Normalize zkevm-benchmark-workload metrics into gasfit inputs",
    )
    prepare_zkevm_p.add_argument("--zkevm-metrics", required=True, type=Path)
    prepare_zkevm_p.add_argument("--out", required=True, type=Path)
    prepare_zkevm_p.set_defaults(func=_prepare_zkevm)
    return parser


def _run(args: argparse.Namespace) -> int:
    fit = GasFit.from_config(Path(args.config))
    fit.load_runtimes(Path(args.runtimes))
    fit.load_opcounts(Path(args.opcounts))
    if args.manifest is not None:
        fit.load_manifest(Path(args.manifest))
    fit.estimate_models()
    if fit.config.glue_adjustment.enabled:
        fit.estimate_glue()
    fit.build_proposal()
    fit.write_reports(Path(args.out))
    return 0


def _compare_campaigns(args: argparse.Namespace) -> int:
    from evm_gasfit.campaign import compare_campaigns

    summary = compare_campaigns(
        Path(args.baseline), Path(args.candidate), Path(args.out)
    )
    print(f"verdict: {summary['verdict']}; {summary['rows']} parameter row(s) compared")
    return 0


def _prepare_eest(args: argparse.Namespace) -> int:
    prepare_eest(Path(args.eest_fixtures), Path(args.out))
    return 0


def _prepare_zkevm(args: argparse.Namespace) -> int:
    prepare_zkevm(Path(args.zkevm_metrics), Path(args.out))
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional override of ``sys.argv[1:]`` for tests.

    Returns:
        Process exit code (0 success, 1 config/input error, 2 modeling error).
    """
    _install_logging()
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse already wrote its error to stderr.
        return int(exc.code) if isinstance(exc.code, int) else 1

    try:
        return args.func(args)
    except ConfigError as exc:
        _log.error("config error: %s", exc)
        return 1
    except ModelingError as exc:
        _log.error("modeling error: %s", exc)
        return 2
    except Exception:  # noqa: BLE001 -- top-level CLI catch-all, logs full traceback below
        _log.error("unexpected error:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
