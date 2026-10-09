# Robot-arm FPGA accelerator

Target board: AMD Kria KR260 Robotics Starter Kit.

Status updated **October 8, 2026**: the KR260 has arrived. A teammate/friend
is preparing its Ubuntu microSD card. First boot and the installed image
version have not been confirmed; M0 hardware acceptance has not run.
M1 remains gated on M0 acceptance on the actual board.

See [current FPGA status](board/status.md) for owners, verification evidence,
open review links and next steps. Existing M0 preparation is tracked in
[PR #18](https://github.com/psamin/robotfpga/pull/18) (runbook/version pins)
and [PR #20](https://github.com/psamin/robotfpga/pull/20) (inventory collector).

Standalone RTL preparation is verified and pushed for review:

- [Signed INT8 MAC, PR #60](https://github.com/psamin/robotfpga/pull/60):
  338,736 XSim cycle checks passed; K26 standalone synthesis passed.
- [Independent MAC lanes, PR #62](https://github.com/psamin/robotfpga/pull/62):
  3,278,586 lane checks passed across 1, 3 and 8 lanes; synthesis passed.
- [Reproducible synthesis runner, PR #66](https://github.com/psamin/robotfpga/pull/66):
  both K26 tops passed with the clock constraint applied before synthesis.

These PRs are awaiting review/merge. These results establish standalone
behavior and synthesis, not placed/routed timing or execution on the board.

The neural-network architecture, 27,658-byte input packet, 48-byte output
packet, integer arithmetic, and hardware/software interface are fixed by the
project specification. Once supplied, `ref/intref.py` and the golden vectors
are the ultimate inference correctness reference. M0 uses a vendor example;
it does not implement or replace any part of the neural network.

The software teammate owns training, quantization, the integer reference,
weights and golden-vector handoff. The FPGA workstream owns RTL verification,
synthesis and board integration, coordinated with existing HLS owners.
See [the build specification](plans/build-spec.md) and
[contribution rules](CONTRIBUTING.md). Assigned issues claim work; each PR
references its issue with `Closes #...`. Publish work branches early and
push verified changes promptly; an open PR is not a merged milestone.

Commit authored source, scripts, documentation, and reviewed dependency locks.
Keep downloaded overlays, installers, generated tool projects, and raw board
logs outside tracked source. Reviewed milestone evidence can be committed
under `results/m0/` after removing machine-specific information.
