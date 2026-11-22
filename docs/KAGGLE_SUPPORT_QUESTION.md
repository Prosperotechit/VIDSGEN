# Question for Kaggle Support (draft, not yet sent)

**Subject:** Is a short, authenticated, temporary tunnel to a notebook allowed for personal ML inference testing?

Hello,

I am testing an open-source video-generation model (Wan 2.2 5B, Apache-2.0) for **personal, non-commercial** research. I would like to run
the model in a Kaggle GPU notebook and, for a **single supervised test of at most 30 minutes**, let my own development machine send a
few requests to it over an **outbound-initiated, authenticated, encrypted tunnel** (the notebook would make only outbound connections;
a bearer-token proxy would be the only thing reachable; I would stop it at the end of the test).

Is this permitted under the Kaggle Terms of Use and Acceptable Use Policy? If not, is there a supported way to exchange data with a
running notebook from my own machine for ML experimentation?

Thank you.

*(Facts to include if asked: personal account; no third-party users; no commercial use; no persistent server; GPU quota only; no
traffic other than my own test requests; the test stops itself after 30 minutes.)*
