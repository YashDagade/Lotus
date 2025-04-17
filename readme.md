# 🧬 Cas9-FlowGen: Generative Modeling of Cas9 Proteins via Latent Flow Matching

## 🧠 Overview
This project aims to develop a generative model of Cas9 proteins by leveraging pretrained protein language models and flow matching in the latent space. Our goal is to build a controllable and efficient framework for designing new Cas9 variants, optionally conditioned on functional motifs such as PAM preferences. Inspired by the ProtFlow architecture, we aim to enable single-step generation of high-quality, biologically meaningful Cas9 sequences for use in genome editing applications.

## 🧪 Motivation
Cas9 enzymes are the cornerstone of CRISPR-based gene editing systems. However, most current applications rely on a few well-characterized variants like SpCas9 or SaCas9. Designing new Cas9 orthologs with diverse PAM compatibility, improved efficiency, and low off-target activity remains a major bottleneck in expanding the CRISPR toolkit.

Existing methods like autoregressive generation and diffusion-based models are limited by:

- Large sequence modeling space
- Slow inference and sampling
- Poor global semantic coherence
- Need for massive compute

We propose a flow-based generative model that operates on a latent representation space extracted from pretrained models (like ESM-2), enabling:

- Fast, single-step sampling
- Semantic preservation
- Conditional generation

Will be updated soon with more details :D