//! # spikenaut-ingest
//!
//! Multi-chain blockchain ingest with state-space interpolation for SNN supervisors.
//!
//! ## The Problem
//!
//! Blockchain data arrives at wildly different rates:
//! - Dynex miner stats:  ~1 Hz
//! - Qubic ticks:        ~0.2–0.5 Hz (2–5 second intervals)
//! - Quai blocks:        ~0.08 Hz (12-second block time)
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
//! ## Provenance
//!
//! Extracted from Eagle-Lander, a private neuromorphic GPU supervisor (closed-source).
//! The interpolation bank ran in
//! production feeding 12-channel blockchain telemetry into a 65,536-neuron LSM
//! at 10 Hz before being open-sourced as a standalone crate.
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
