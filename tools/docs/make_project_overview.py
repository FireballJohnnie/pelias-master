# Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
# provenance: original
# Builds docs/Pelias_project_overview.pdf: the project in one document (2026-10-30).
#   python3 tools/docs/make_project_overview.py OUT.pdf ["TIER RESULT TEXT"]
# Every number is copied with its label from docs/PELIAS_RESIDUAL_PLANES.md, docs/PELIAS_NEW_METHODS.md,
# docs/PELIAS_SYSTEM_MASTERWORK.md (sections 20-21), the pathways document v2 and logs/BUILD_LOG.md. Nothing new is measured here.
import sys
from pelias_pdf_common import *

OUT = sys.argv[1]
s = []
s += [P("Pelias", TITLE),
      P("One consumer PC running a 35-billion-parameter model as if CPU and GPU were one machine, and reading precision only where it "
        "matters: how it was found, what is measured, what was ruled out, and what the same hardware ideas may offer physics and "
        "the other sciences", SUB),
      P("John Lipka and the Pelias contributors · 30 October 2026 · project overview (pelias-master, docs/)", SM),
      Spacer(1, 6),
      note("<b>How to read the numbers.</b> Everything labelled MEASURED was measured on one machine: an RTX 4070 (12 GB, memory overclocked "
           "and disclosed), a Ryzen 7 5800X3D, 64 GB of DDR4-3200, on the Qwen3.6-35B-A3B mixture-of-experts model. DERIVED means worked out "
           "on paper from measured inputs, with the formula written down; PLAN means designed but not yet built or run; PUBLISHED means "
           "another team's number. On any other machine every figure is a proposal to benchmark, not a result. Accuracy is scored as the "
           "Kullback-Leibler divergence (KL) of the model's next-token distribution from the original full-precision model's, token by "
           "token over 25,500 tokens of WikiText-2, with paired 95 % intervals; lower is better and 0 is a perfect copy."),
      Spacer(1, 4)]

s += [P("1. In one page", H1),
      P("A large language model of the mixture-of-experts kind keeps 256 small sub-networks (\"experts\") in each of its 40 layers and "
        "uses only 8 of them for any one word. The whole model is 71 GB at 16-bit precision; compressed to about 4.5 bits per weight it "
        "is 22.65 GB, still nearly twice the 12 GB of memory on the graphics card. The usual answer is a larger card or a data centre. "
        "Pelias treats the processor, the graphics card and their shared memory as one system instead: the experts used most stay on the "
        "card, the rest are computed by the processor at the same time from main memory, and the two hand work to each other in about "
        "2 microseconds instead of 30."),
      P("Compression costs accuracy, and the cost shows up where the model is torn between two nearly equal next words. Over the last two "
        "days the project found a way to buy most of that accuracy back cheaply: store the small difference between the compressed and the "
        "original weights as one or two extra sign bits per weight with a scale per 256 weights (\"residual sign planes\"), and read them "
        "only where and when they matter. With two planes the error against the original model fell by 55 % (MEASURED, settled), and at "
        "the near-ties the original model sided with Pelias 12 times out of 15 where it disagreed with the reference engine, llama.cpp "
        "(MEASURED). The newest step gives each of the 10,240 experts its own precision according to how often it is used, at exactly the "
        "size of the best-known public file, which the standard file format cannot express: measured, it cuts the error by 26.6 % at the same size."),
      table(["Result", "Value", "Label"], [
          ["Model file larger than the card's memory, running interactively", "1.69x the card's memory; 37.9 to 54.1 tokens/s", "MEASURED"],
          ["CPU-to-GPU hand-off", "2.40 µs (\"doorbell\") vs 30.5 to 34.0 µs launch-and-wait", "MEASURED"],
          ["CPU and GPU reading the same RAM together", "47.62 GB/s, 93 % of the 51.2 GB/s peak", "MEASURED"],
          ["Best interim engine vs best stock layout", "53.5 vs 41.1 tokens/s (+30 %)", "MEASURED"],
          ["Two residual planes on the public 4-bit file", "KL 0.012808 -> 0.005818 (-54.6 %, t -12.25)", "MEASURED"],
          ["Near-tie disagreements judged by the original model", "Pelias 12, llama.cpp 3", "MEASURED"],
          ["Usage-tiered experts at the public file's exact size", "KL 0.012808 -> 0.009404 (-26.6 %, t -12.41) on WikiText; about 0.0110 on chat and code",
           "MEASURED; chat and code DERIVED"],
      ], [3.4, 2.9, 1.2])]

s += [P("2. How it came to be: an accident, three times over", H1),
      P("In the author's words (26 October 2026): <i>\"this started as an effort for me to better create my own llm, when i found this... "
        "accidentally. everything said we couldn't train an moe using my type of system, we found a way. then we saw the hardware ceiling "
        "limit when running that quantized llm, and we found a way to unify the bus to make it run even better from the way we found to "
        "train it... and then even more!\"</i>"),
      *bullets([
          "<b>Training first (September to mid-October 2026).</b> The goal was a personal assistant model. Fine-tuning a 35B mixture-of-experts "
          "model on one 12 GB card was described everywhere as impossible. Streaming each layer's experts through the card once per step, with "
          "every micro-batch passing before the next layer, cut the traffic per training step from 121.5 to 35.8 GiB (DERIVED) and made it run.",
          "<b>Then the inference ceiling (20 to 23 October).</b> Running the trained, compressed model showed the processor and the card taking "
          "turns, each idle while the other worked: the engine's own timers found zero CPU/GPU overlap and 4.6 to 5.0 ms of hand-off per token "
          "(MEASURED). Treating the two processors and the memory between them as one bus came from measuring those gaps.",
          "<b>Then accuracy (28 to 30 October).</b> Comparing word-by-word output against the reference engine showed that disagreements cluster "
          "at near-ties. Asking the original model who was right, instead of assuming the reference engine was, turned the question from "
          "\"match llama.cpp\" into \"match the original\", and that is where the residual planes came from.",
          "<b>The rule throughout:</b> the code is written from the ground up in C++20 and CUDA, with no third-party inference code; existing "
          "engines are read as specifications and used only as test references. Each idea is kept only if a paired measurement on the machine "
          "shows a gain."])]

s += [P("3. The unified bus", H1),
      P("A token passes through 40 layers. In each, the GPU runs the attention and the experts it holds in its memory (the \"hot bank\"), "
        "while the CPU runs the other selected experts straight from main memory; the results meet on the card. What makes this one system "
        "rather than two machines taking turns are a handful of measured primitives:"),
      table(["Primitive", "What it does", "Measured"], [
          ["Hot bank", "The most-used experts stay in video memory, chosen by a water-fill solver over measured usage", "Byte hit rate 0.775 to 0.779 live, 7.0 GiB bank"],
          ["Doorbell", "CPU and GPU signal each other through a flag in memory, a resident GPU thread waiting on it", "2.40 µs round trip vs 30.5 µs (Windows) / 34.0 µs (WSL2) launch and wait: 12.7x to 14.2x"],
          ["Dual bus", "CPU and GPU read the same DDR4 at once", "26.56 + 21.06 = 47.62 GB/s, 93 % of peak"],
          ["Graph replay + flush rule", "A whole step's GPU work recorded once and replayed; copy and compute streams run together", "0.716 µs per kernel vs 4.70 µs; overlap 0.92 of ideal vs 0.47"],
          ["One streaming kernel per token (R3)", "One persistent GPU kernel streams every weight of a token in one long read", "8.76 -> 5.685 ms per token; pure streaming about 4.75 ms at about 478 GB/s"],
          ["Integer tensor path", "4-bit weights unpacked in registers, 8-bit activations, quantization fused into the norm", "1.39x to 1.84x the 16-bit path on all 20 real shapes"],
      ], [1.55, 3.1, 2.85]),
      Spacer(1, 3),
      P("The limits are stated plainly in the project records: the RAM's own peak (51.2 GB/s here) does not grow, and the two readers "
        "already reach 93 % of it. What the bus changes is how much useful work each trip to memory does and when the trips happen. "
        "The full Pelias engine has not yet run end to end; the 53.5 tokens/s comes from the interim build.")]

s += [P("4. The 7-bit + 1 system: storing fewer bits and reading them exactly", H1),
      P("Weights are stored as 8-bit integers in a way that lets the eighth bit be stored apart, skipped, or added back later, exactly:"),
      *bullets([
          "<b>Zip scales (Q8_Z).</b> One 16-bit scale per 256 weights plus a 6-bit level per 32, instead of a 16-bit scale per 32: same accuracy "
          "as the standard 8-bit format at 8.25 instead of 8.5 bits per weight (MEASURED, whole model, no difference).",
          "<b>The eighth bit as a gate (Q7_ZT, 7.53 bits per weight).</b> Each 8-bit value is a 7-bit base plus one \"light\" bit H that refines "
          "one step and never sets the range. H is kept only on the 2 of every 8 groups of 32 weights where it removes the most error. Whole "
          "model: no difference from 8 bits, settled better than pure 7 bits (t -3.54), and 236 MB smaller than the best public file with "
          "lower drift (not settled) (MEASURED). If H is lost, a light bit costs 1.4x the error; a range bit would cost about 360x.",
          "<b>The 1-bit decoder (T-decode).</b> The eighth bit is added back with AND and popcount against the 8 bit planes of the "
          "activation, with no multiplications, on units that are otherwise idle: integer-exact (unit tests); the GPU's 1-bit tensor path "
          "runs at 937.4 trillion bit-operations per second, 8.00x its 8-bit rate (MEASURED). A missing bit replaced by its mean (0.5) "
          "through one sum per block costs 43 % less than any 0/1 guess and stays exact.",
          "<b>Split storage.</b> The base and the H bits live in separate arrays, so the base can sit in video memory and H in main memory, "
          "or on another unit, and be merged exactly (unit-tested)."]),
      P("This is the bit-level counterpart of the bus: the precision of a number is no longer one block that must sit in one place, but "
        "layers that can live where the bandwidth is and be read when they are needed.")]

s += [P("5. The new accuracy method: residual planes and usage-tiered experts", H1),
      P("<b>Residual sign planes.</b> For every 256 weights the difference r between the original and the stored value is approximated as "
        "s<sub>1</sub>b<sub>1</sub> + s<sub>2</sub>b<sub>2</sub> + ..., each b a vector of signs (+1/-1, one bit per weight) and each s one "
        "16-bit scale, fitted plane by plane and then jointly (least-squares scales, best sign pattern per weight). Each plane removes about "
        "two thirds of what is left (MEASURED on real weights). The planes are a separate file, so any existing compressed model can be "
        "improved without touching it, and a reader can take 0, 1 or 2 planes per token."),
      table(["Test (public 4-bit file, 22.65 GB)", "KL", "Same top word", "Label"], [
          ["The file alone, through Pelias", "0.012808", "95.02 %", "MEASURED"],
          ["+ 2 planes (10.2 GB side file)", "0.005818 (-54.6 %, t -12.25)", "96.77 %", "MEASURED"],
          ["+ 2 planes + exact top-64 re-score of the output layer", "0.005741", "96.86 %", "MEASURED"],
          ["Near-tie splits judged by the original model", "Pelias 12, llama.cpp 3 (without planes 7 vs 4)", "-", "MEASURED"],
      ], [3.0, 2.35, 1.0, 1.15]),
      Spacer(1, 4),
      P("<b>The honest catch, and what it led to.</b> Bit for bit, a plane is not cheaper than simply storing one more bit up front "
        "(MEASURED). So the question became where to spend bits. Expert use is very uneven, but it depends on the text: the 64 experts per "
        "layer most used on encyclopedia text carry 72 % of its routing but only 36 % of chat and code routing (MEASURED). A standard file "
        "gives all 256 experts of a layer one type. Pelias now stores each expert in its own type (a 4-bit floor of our own design, Q4_TG2, "
        "or 5, 6 or 8 bits), chosen by usage across both kinds of text and by the error each layer's experts show when passed through their "
        "own nonlinearity, measured on experts from all 40 layers of the original model."),
      table(["Option (WikiText / chat+code)", "Size", "KL", "Extra video memory", "Label"], [
          ["Public 4-bit file", "22.65 GB", "0.012808 / about 0.0129", "-", "MEASURED / DERIVED"],
          ["Usage-tiered experts", "22.65 GB", "<b>0.009404</b> (-26.6 %, t -12.41) / about 0.0110", "+0.71 GB for the hot experts", "MEASURED / DERIVED"],
          ["+ 1 GB of planes on the worst-error blocks, applied by the CPU", "23.65 GB", "about 0.0088 / 0.0100", "none", "DERIVED"],
          ["+ 2 GB of such planes", "24.65 GB", "about 0.0083 / 0.0092", "none", "DERIVED"],
          ["+ 2 GB, read only for each token's 4 most-weighted experts", "24.65 GB", "about 0.0085 / 0.0096 at half the plane reads", "none", "DERIVED"],
      ], [2.6, 0.8, 1.5, 1.4, 1.2]),
      Spacer(1, 3),
      P("The tiered file was built from the original model on the PC in 34 minutes, 22,663,406,784 bytes against the public file's "
        "22,663,387,424, with exactly the predicted mix of types, and measured: KL 0.009404 against 0.012808 (-26.6 %, 95 % interval "
        "[-0.003949, -0.002860], t -12.41), lower on 94 of 100 text chunks, and the largest cut (-30.4 %) on the hardest stretch of text "
        "(MEASURED). It had been predicted at 0.0089 (range 0.0080 to 0.0100). The remaining rows are projected through the three measured "
        "points (error left -> KL: 1.0 -> 0.012808, 0.484 -> 0.009404, 0.098 -> 0.005818), so each is a prediction until its run; chat-and-code "
        "KL has no measured baseline yet.")]

s += [P("<b>How a user gets this: three ways</b> (each a choice in the Pelias interface)", H2),
      table(["Way", "What the user needs", "What changes", "Accuracy here", "Runs in"], [
          ["1. A premade file as is", "The file (Unsloth's or anyone's)", "Nothing", "KL 0.012808 (MEASURED)", "Any engine"],
          ["2. Improve that file", "A plane side file for that exact file: downloaded ready-made, or built once from the original weights",
           "Nothing in the file; the side file sits next to it and is refused on any other file", "2 planes: 0.005818 (MEASURED); 1-2 GB planes on a stock file UNMEASURED",
           "The file anywhere; the planes in Pelias"],
          ["3. A Pelias tiered quant", "The original weights once; about an hour of PC time at 35B", "The experts rebuilt by use, same size; planes optional on top",
           "0.009404 at the same size (MEASURED)", "Pelias only"],
      ], [1.25, 1.85, 1.75, 1.55, 1.1]),
      Spacer(1, 3),
      P("The planes store the gap between a file and the original weights, so way 2 needs the original only once: whoever has it builds the "
        "side file, and anyone with the same file can use it, a 1 to 2 GB download instead of 71 GB. Way 2 keeps the file portable; way 3 is "
        "the most accurate per byte.")]

s += [P("6. Why the unified bus makes all of this work together", H1),
      *bullets([
          "<b>Precision lives where the bandwidth is.</b> Planes and the eighth bit can sit in main memory and be applied by the CPU while the GPU "
          "runs the base: one plane read on every token adds about 31 to 63 MB per token on the CPU's bus and nothing to video memory "
          "(DERIVED). That bus is already the busier side when the hot experts miss, so the planes cost about 0.65 to 1.3 ms there; reading "
          "them only for each token's 4 most-weighted experts keeps 81 % of their benefit at half the reads (the router weights fall from 0.245 "
          "to 0.074 by rank, MEASURED), and the hot experts' planes can move to video memory; the engine will measure it.",
          "<b>Idle units do the extra work.</b> Sign planes and the gated bit are AND/popcount work, the kind that runs on integer and 1-bit "
          "units that sit idle during decoding (the GPU drew 45 to 47 W of its 200 W while reading RAM, MEASURED).",
          "<b>Usage decides placement and precision together.</b> The same routing statistics that pick the hot bank pick which experts get finer "
          "storage; the bus then keeps those in the fastest memory.",
          "<b>Hand-offs are cheap enough to do per layer.</b> A correction that must reach the GPU before a nonlinearity is about 80 hand-offs per "
          "token; at 1 to 2.4 µs each that is affordable, at 30 µs it would not be."])]

s += [P("7. What was ruled out (kept so nobody retries them)", H1),
      table(["Idea", "Result", "Label"], [
          ["Pure 7-bit weights everywhere", "+0.001122 KL vs 8-bit, settled worse", "MEASURED"],
          ["A range (\"heavy\") eighth bit", "Ties when read, about 360x worse when missing", "MEASURED"],
          ["Deferring or swapping experts at near-ties", "+0.0042 and +0.0017, settled worse", "MEASURED"],
          ["Rotating the routed experts (rotation helps the dense part)", "No gain", "MEASURED"],
          ["Ray-tracing cores as a prefetch map; video units for weights or cache", "Worse than simple rules; 1.3 s per token", "MEASURED"],
          ["Lossless entropy coding of active 4-bit weights", "Only 3 to 4 % smaller: codes are near full entropy already", "MEASURED"],
          ["Full precision inputs to fix near-ties", "Nothing gained: the lever is the weights", "MEASURED"],
          ["Uniform planes as a size-for-size win", "One more stored bit beats one plane per byte", "MEASURED"],
          ["One usage ranking for every kind of text", "36 % coverage on chat and code vs 72 % on its own text", "MEASURED"],
          ["A per-expert output scale to undo shrinkage", "Removes 0.00 to 0.06 % of the error: the shrink is noise", "MEASURED"],
      ], [3.4, 3.0, 1.1])]

s += [P("8. What exists elsewhere, and what we could not find", H1),
      P("Most single pieces exist in published work. Running cold experts on the CPU while hot ones stay on the GPU is done by PowerInfer, "
        "KTransformers and llama.cpp's --n-cpu-moe. Sign-bit planes are a known family (binary-coded quantization: AnyBCQ, Any-Precision LLM, "
        "BitStack). Per-expert precision is studied offline (MoPEQ; GEMQ, ICML 2026, with a Fisher-based global allocation) and at run time "
        "(DynaExq; DPA-MoE, October 2026, which reads a varying number of sign planes per expert from the router's scores, as a llama.cpp "
        "prototype with everything resident in GPU memory). Bit-serial integer arithmetic is used by T-MAC and others."),
      P("What our searches did not find is one system that combines them across both memory buses: precision split into layers that live "
        "in different memories, read per token by whichever unit has idle capacity, on top of an expert placement solved for the same "
        "machine, and checked against the original model rather than against another engine. \"Not found\" means not found by our searches, "
        "not a claim that nobody has done it.")]

s += [P("9. Beyond language models", H1),
      P("Nothing in the bus is specific to language models: each primitive was measured as a hardware property by stand-alone programs "
        "before any model ran on it, and again through Direct3D 12 and Vulkan. A gain carries over to another field <b>only if</b> that "
        "field's work has the same bottleneck; every row is a hypothesis with its condition and its test (docs/PELIAS_SYSTEM_MASTERWORK.md "
        "section 21 has the full table and the Amdahl arithmetic). End to end, gain = 1 / ((1 - f) + f / k) - 1, with f the share of the run "
        "the primitive fixes; for the language model f was back-solved at 27 % to 52 % (DERIVED)."),
      table(["Field", "What maps onto the bus", "Condition", "Factor from this project"], [
          ["Recommendation, vector search and retrieval", "Hot rows or clusters in video memory, cold ones read by the CPU at once", "Skewed access, transfer-bound", "1 / (1 - h) less host traffic: 5.8x at h = 0.827"],
          ["Scientific simulation (CFD, FEA, out-of-core solvers, lattice QCD)", "Sweeps reordered so each block loads once per pass; low-precision base plus corrections", "Transfer-bound, reorderable sweeps", "3.4x less traffic in our loop; dual readers 1.82x over the GPU alone"],
          ["Monte Carlo particle transport", "Most-touched cross-section data on the card, the rest looked up by the CPU", "Lookup-bound and skewed", "1 / (1 - h); hand-off 12.7x to 14.2x"],
          ["Detector, video and other multi-stage pipelines; control loops", "CPU and GPU stages exchanging results per event, frame or cycle", "Many fine-grained hand-offs", "Doorbell 2.0 to 2.4 µs vs 30.5 µs; graph replay 3.1x to 6.6x lower launch cost"],
          ["Extended-precision linear algebra", "Exact integer slices on 1-bit and integer tensor units (Ozaki-scheme family)", "GEMM-bound, precision can be sliced", "1-bit path 937.4 T bit-op/s, 8.00x int8"],
          ["Stored scientific data", "Coarse base plus sign-plane side files read on demand", "Uneven use, coarse first pass acceptable", "Each plane removes about 2/3 of the remaining error (on weights)"],
          ["Games and graphics", "Doorbell in video memory, replayed command lists, CPU writes straight into VRAM", "Sync points, submissions or streaming on the critical path", "42x to 78x per sync point; 36x to 57x per submission; 1.78x per upload (P-G0)"],
          ["Home and small-office servers", "Working sets above the card's memory at interactive speed", "VRAM size, not compute, is the limit", "1.69x the card's memory measured here"],
      ], [1.6, 2.4, 1.6, 1.9]),
      Spacer(1, 4),
      P("The same accuracy ideas have numerical-physics counterparts: residual planes resemble iterative refinement and mixed-precision "
        "solvers; exact bit-plane sums resemble the Ozaki scheme; usage-tiered precision resembles adaptive refinement with a floor; "
        "worst-error block planes resemble a-posteriori error indicators.")]

s += [P("10. Where it stands and what comes next", H1),
      *bullets([
          "<b>Done (30 October, 08:16):</b> the usage-tiered file at the public file's exact size, scored on 100 chunks against the original "
          "model: KL 0.009404 against 0.012808 (-26.6 %, settled), same top word 95.80 % against 95.02 % (MEASURED).",
          "<b>Next:</b> the worst-error plane side files (1 GB and 2 GB) on the tiered base, then the split-token grade against the original "
          "model; then the engine itself (one token loop on both processors), where end-to-end speed is measured.",
          "<b>Open measurements:</b> per-layer sensitivity (four runs), a chat-and-code baseline of the original model, energy per token, and the "
          "same tests on a second machine.",
          "<b>Not claimed:</b> any result on hardware other than the one machine; any non-language-model gain until its own probe runs."])]

s += [P("Sources", H1),
      P("Project records: docs/PELIAS_RESIDUAL_PLANES.md, PELIAS_NEW_METHODS.md, PELIAS_7_T_H_PLAN.md, PELIAS_SYSTEM_MASTERWORK.md (sections "
        "10, 20, 21), PELIAS_GAME_PLAN.md and logs/BUILD_LOG.md in this repository. Reproduce: pelias planes, pelias quantize --tier "
        "(tools/tier_alloc.py), pelias kld, pelias teacher with tools/split_grade.py (README).", SM),
      P("Published work: AnyBCQ, arXiv:2510.10467; Any-Precision LLM (Park et al., ICML 2024), arXiv:2402.10517; BitStack, arXiv:2410.23918; "
        "DynaExq, arXiv:2511.15015; MoPEQ, arXiv:2509.02512; GEMQ (ICML 2026), arXiv:2605.23078; DPA-MoE, arXiv:2610.21378; Unsloth's Qwen3.5 GGUF "
        "benchmarks (unsloth.ai); Clark et al., mixed-precision lattice QCD solvers on GPUs, arXiv:0911.3191; DGEMM emulation with the Ozaki "
        "scheme on FP8/INT8 tensor cores, arXiv:2508.00441; performance-portable Monte Carlo particle transport on GPUs (OpenMC), "
        "arXiv:2403.12345; PowerInfer, KTransformers and T-MAC as cited in the project records.", SM)]

build(OUT, "Pelias: project overview", s, "Pelias project overview, 30 October 2026. Measured on one machine; every other figure is a proposal to benchmark.")
print("wrote", OUT)
