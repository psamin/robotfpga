# Briefs for psamin: speed of SmolVLA on the KR260 (2026-10-22)

Two documents, same measurements, different question:

| File | Question it answers |
|---|---|
| `kr260-measured-path-3s-and-continuous.pdf` | **Full analysis and the milestone plan.** Where the 24.4 s goes; the precision cost of every arithmetic option measured on the real checkpoint (49 variants x 3 frames); what each engine costs on the K26 (57 HLS syntheses); model/system levers; the control-loop rule; a plan with a stopping point at ~4-5 s per chunk after one week and the continuation to continuous motion. |
| `kr260-direct-path-to-continuous.pdf` | **Shortest route to a continuously moving arm only** (no intermediate deliverables): the latency requirement (<= 1.1 s per chunk with two inferences in flight), the budget and how each term is met, the parallel work plan with owners and days, what would shorten it, risks. |

Earlier one-page version (estimates only, superseded): `build/psamin-options-3s-per-chunk.pdf` on the FPGA laptop.
Measurements and scripts behind both: `build/brief/` on the FPGA laptop (precision_results.md, hardware_options.md,
model_system_levers.md, costmodel.py, the HLS projects). `src/` holds the generators of the two PDFs (reportlab).
