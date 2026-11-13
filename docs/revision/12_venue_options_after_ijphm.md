# Deliverable 12: Venue options after the IJPHM desk rejection

Checked 2026-11-13. "Verified" = read on the venue's or publisher's own page in this session. "Unverified" = from listings, aggregators or earlier notes; check before relying on it.

## What happened at IJPHM

- Submitted 2026-11-09 (ID 5305). Desk-rejected about 25 minutes later, without peer review.
- The email said: "we do not accept submissions of a revised manuscript based on this rejected article." That stops a resubmission **to IJPHM only**. It does not limit any other venue.
- Most likely cause: scope. IJPHM is about machine health (faults, wear, remaining useful life). Three of the four datasets (SWaT, WADI, HAI) are cyber-attack datasets and the abstract talks about attacks, so it read as an industrial-security paper. Other likely factors: the introduction tells the story of the first version's mistakes; 20 pages against IJPHM's suggested 10 to 15; the note to the editor mentioned the ICACS rejection.

## Options

| Venue | Type | Fit | Cost to publish | Length | Timing | Status of facts |
|---|---|---|---|---|---|---|
| **IFAC SAFEPROCESS 2027** (13th IFAC Symposium on Fault Detection, Supervision and Safety for Technical Processes), Delft, NL, 29 Jun to 2 Jul 2027 | Conference | **Strong for both angles.** Official topic list includes data-driven diagnosis, anomaly detection, condition and health monitoring, and cybersecurity for cyber-physical systems. | Registration fee to present (amount not checked) | IFAC papers are usually 6 pages (not checked for 2027) | Regular papers due **22 Nov 2026** (extended from 31 Oct); notification 28 Feb 2027 | Topics verified (ifac-control.org). Deadline extension from the official conference site's listing. Notification date from WikiCFP. |
| **International Journal of Critical Infrastructure Protection** (IJCIP, Elsevier) | Journal | **Strong.** Scope is critical-infrastructure protection, naming drinking water and water treatment, chemicals and critical manufacturing. | **Free** on the subscription route; open access costs USD 3,110 | Research papers 5,000 to 10,000 words incl. abstract, excl. references | Rolling; review time not stated | Verified (Elsevier guide for authors and open-access page). Earlier SWaT papers in IJCIP: unverified. |
| **Journal of Information Security and Applications** (JISA, Elsevier) | Journal | Medium. Applied information security; less about physical processes. | Hybrid; open-access fee not checked | Not stated | Rolling | Partly verified. Uses author-year references, same as the current draft. |
| **SN Computer Science** (Springer) | Journal | Broad, so fit is not a problem | Hybrid; subscription route normally free | Not checked | Reports of slow reviews (5+ months, multiple rounds) | Model verified; timing from forum anecdotes only. |
| **International Journal of Information Technology** (Springer / BVICAM) | Journal | Broad IT | Subscription route free **only up to 6 typeset pages**, then ₹2,000 per extra page for Indian authors; open access USD 3,390 | See fee rule | Rolling | Verified on BVICAM's journal page. At ~20 pages this could cost about ₹28,000. |

**Not recommended** (from the 2026-11-09 session, aggregator sources): Expert Systems with Applications (about 11% acceptance), Engineering Applications of AI (desk-rejects work without a new AI method), Computers & Security (very selective), MDPI Sensors and IEEE Access (article fees).

## Rules that apply whichever is chosen

- **One venue at a time.** Never submit the same paper to two places at once.
- **Conference first, journal later is normal**, but the journal version must be substantially extended, and both submissions must say so. Check the journal's policy on prior conference versions before doing this.
- **Preprints:** IJCIP and JISA (Elsevier) say an arXiv preprint does not count as prior publication. IFAC's preprint policy was not checked. First-time arXiv authors need an endorsement.
- **Cover letter:** no need to mention the earlier rejections. If a form asks directly, answer truthfully in one line.

## Changes needed before any resubmission (no new experiments)

1. Reframe for the chosen venue. For IJCIP or SAFEPROCESS: lead with industrial-control-system and critical-infrastructure monitoring, SWaT/WADI/HAI first.
2. Remove the "our first analysis was wrong" story from the introduction; keep the SWaT re-upload issue as a finding about the dataset.
3. Use the venue's template and reference style (IJCIP: numbered; IFAC: its own template), and fit its length (SAFEPROCESS would mean cutting to about 6 pages).
4. Update README.md, CHECKPOINT.md and CLAUDE.md, which still name IJPHM as the target.

## Decision needed (Sarjan and Prof. Shah)

- **A. SAFEPROCESS 2027 now** (deadline 22 Nov, 9 days): a 6-page version; a longer journal version can follow later.
- **B. IJCIP**: full reframed paper, no deadline, free on the subscription route.
- **C. A broader journal** (SN Computer Science) if acceptance odds matter more than venue strength; expect slow reviews.

## Added 2026-11-14: free venues with fairer odds (Sarjan wants reputable, free, not too hard)

No journal publishes an acceptance rate on its own page, so "easier" below is a judgement, not a number.

| Venue | Why it fits the brief | Cost | Speed | Facts |
|---|---|---|---|---|
| **TMLR** (Transactions on Machine Learning Research, JMLR family) | Judges whether claims are supported and whether someone would be interested; states novelty is **not** required. Suits a careful evaluation study, and novelty was the weak score at ICACS. | No fees listed on its pages (widely reported as free; confirm) | Aims for a decision in about 9 weeks | Criteria and timing verified (jmlr.org/tmlr). Allows an arXiv preprint. Per-author yearly submission quotas apply. |
| **Sādhanā** (Indian Academy of Sciences, Springer) | Respected Indian engineering journal; scope includes computer science; SCIE and Scopus indexed; impact factor 1.6 (2025). | Hybrid; subscription route usually free, but author fees not stated on the page (confirm) | Median 57 days to first decision | Verified on Springer's journal page. |
| **Journal of The Institution of Engineers (India): Series B** (Springer) | Scope includes computer engineering; Indian-run. | Hybrid; fees not stated (confirm) | Median 76 days to first decision | Verified on Springer's journal page. |
| **SN Computer Science** (Springer) | Broad, Scopus and DBLP indexed. | Hybrid; fees not stated (confirm) | Median 96 days to first decision | Verified on Springer's journal page. |
| DMLR (Data-centric ML Research, JMLR family) | Scope names benchmark and evaluation methodology; no author fees. | Free (verified) | Not stated | States a "high bar", so not an easy option. |

**arXiv** is a free preprint server, not peer-reviewed publication. It gives a public, dated link. TMLR, IJCIP and JISA allow it. First-time authors need an endorsement.
**IEEE** is a publisher: its journals are selective, IEEE Access charges an article fee, and IEEE conferences charge registration to publish in IEEE Xplore.
