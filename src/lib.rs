//! # spikenaut-ingest
//!
//! Multi-chain blockchain data ingest with state-space interpolation for SNN supervisors.
//!
//! ## The Problem
//!
//! Blockchain data arrives at wildly different rates:
//! - Dynex data:  ~1 Hz
//! - Qubic data:  ~0.2–0.5 Hz (2–5 second intervals)
//! - Quai data:   ~0.08 Hz (12-second intervals)
//!
//! An SNN supervisor running at 10 Hz sees identical values for 20–120 steps,
//! then a sharp discontinuity — creating phantom spikes that drown real signal.
//!
//! ## The Solution
//!
//! First-order exponential state-space interpolation:
//! ```text
//! x[k+1] = α · x[k] + (1 - α) · u[k]
//! ```
//! where `α = exp(-Δt / τ)` and `τ` is tuned per signal class.
//!
//! ## Core Responsibilities
//!
//! - **Acquire raw inputs** from sensors, system telemetry, and optional external feeds.
//! - **Normalize & sanitize** values into stable numeric ranges.
//! - **Map named fields** into a deterministic channel vector (`[f32; N]` ABI).
//! - **Buffer & align** samples in time (timestamps, sample rates, windows).
//! - **Export snapshots** in a compact, testable format for encoders & simulators.
//! - **Provide utilities** such as migration helpers, golden fixtures, and CI smoke examples.
//!
//! ## Why It Matters for SNN / LLM Fusion
//!
//! - **Deterministic inputs** let encoders pre-allocate and map channels to neurons without runtime guessing.
//! - **Normalization** guarantees consistent value ranges across machines and datasets.
//! - **Fixed ordering** (the 12-channel layout) acts as a contract every repo can rely on, simplifying fusion and hardware export.
//! - **Separation of concerns** keeps ingestion pure; hardware/ML logic lives in other crates, making each repo easier to review and license-clean.
//!
//! ## Performance Optimizations
//!
//! - **SIMD-ready Struct of Arrays (SoA)** layout in `InterpolatorBank` enables automatic vectorization
//! - **Denormal protection** prevents 10-100x performance degradation when values approach zero
//! - **Zero-allocation design** keeps all state on the stack for cache efficiency
//!
//! ## Provenance
//!
//! Extracted from Eagle-Lander, the author's own private neuromorphic GPU supervisor repository (closed-source).
//! The interpolation bank ran in
//! production feeding 12-channel blockchain telemetry into a 65,536-neuron LSM
//! at 10 Hz before being open-sourced as a standalone crate.
//!
//! ## Architectural Note for spikenaut-synapse
//!
//! While this crate was originally designed for blockchain telemetry (Dynex, Qubic, Quai),
//! the mathematical foundations are valuable for LLM fusion:
//!
//! - The **exponential decay** logic can model SNN membrane potential degradation over time
//! - The **reward tracker** can be repurposed as an "attention reward" system for important LLM tokens
//! - The **interpolation concepts** apply to any time-series data needing smooth 10Hz sampling
//!
//! For pure LLM workloads without blockchain dependencies, consider extracting these mathematical
//! primitives into a more generic crate.
//!
//! ## References
//!
//! - Franklin, G.F., Powell, J.D., & Emami-Naeini, A. (2019).
//!   *Feedback Control of Dynamic Systems* (8th ed.). Pearson.
//!   Zero-Order Hold (ZOH) discretization: α = exp(−Δt/τ).
//!
//! - Kálmán, R.E. (1960). A New Approach to Linear Filtering and Prediction Problems.
//!   *Journal of Basic Engineering*, 82(1), 35–45.
//!   <https://doi.org/10.1115/1.3662552>
//!   Theoretical basis for recursive state estimation.
//!
//! - Ogata, K. (2010). *Modern Control Engineering* (5th ed.). Prentice Hall.
//!   First-order IIR filter as discrete state-space model.
//!
//! ## Usage
//!
//! ```rust
//! use spikenaut_ingest::{ChannelInterpolator, SignalClass};
//!
//! let mut interp = ChannelInterpolator::new(SignalClass::Blockchain);
//!
//! // Feed a new observation from the RPC (irregular cadence)
//! interp.observe(42.0);
//!
//! // Step at 10 Hz to get smooth output
//! let smooth = interp.step();
//! println!("Smoothed value: {}", smooth);
//! ```

pub mod interpolator;
pub mod consensus_reward;
pub mod snapshot;

#[cfg(feature = "async")]
pub mod triple_bridge;

pub use interpolator::{ChannelInterpolator, InterpolatorBank, SignalClass};
pub use consensus_reward::{ConsensusRewardTracker, REWARD_CEILING};
pub use snapshot::TripleSnapshot;
