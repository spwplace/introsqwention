//! Worker command protocol and coordination for tensor-parallel introspection.
//!
//! The master (rank 0) sends [`WorkerCommand`]s over IPC before each model
//! operation. Workers execute the same operation (participating in NCCL
//! collectives) and discard results.

use serde::{Deserialize, Serialize};

/// Commands sent from master to TP workers.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum WorkerCommand {
    /// Run a forward pass with introspection (hidden state capture).
    ForwardIntrospect {
        text: String,
        layers: Option<Vec<usize>>,
    },
    /// Run autoregressive generation.
    Generate {
        text: String,
        max_new_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
    },
    /// Set steering vectors on the model. Each entry is (layer_idx, flat f32 vec).
    SetSteering { vectors: Vec<(usize, Vec<f32>)> },
    /// Set a scaled steering vector across a range of layers.
    SetSteeringRange {
        start_layer: usize,
        end_layer: usize,
        vector: Vec<f32>,
        scale: f64,
    },
    /// Clear all steering vectors.
    ClearSteering,
    /// Set an activation patch (replace hidden state at a layer).
    SetPatch {
        layer_idx: usize,
        hidden_state: Vec<f32>,
    },
    /// Clear all activation patches.
    ClearPatches,
    /// Enable/disable MoE routing capture.
    SetCaptureRouting(bool),
    /// Graceful shutdown.
    Shutdown,
}

// ── Worker side (runs in worker processes) ──────────────────────────

/// IPC socket name for the command channel (master → workers).
#[cfg(feature = "nccl")]
const CMD_SOCKET: &str = "introsqwention_cmd.sock";

/// Worker main loop: receive commands from master, execute on local model.
///
/// This function never returns (except on Shutdown or IPC error).
#[cfg(feature = "nccl")]
pub fn worker_loop(model: mistralrs_core::introspection::IntrospectionModel) -> anyhow::Result<()> {
    use std::collections::HashSet;
    use std::io::{BufRead, BufReader};

    use candle_core::{DType, Tensor};
    use interprocess::local_socket::traits::Stream;
    use interprocess::local_socket::{GenericNamespaced, Stream as LocalStream, ToNsName};

    let ipc_name = CMD_SOCKET.to_ns_name::<GenericNamespaced>()?;

    loop {
        // Connect to master's command listener for each command
        let stream = match LocalStream::connect(ipc_name.clone()) {
            Ok(s) => s,
            Err(e) => {
                // Master may not have the listener up yet; brief retry
                tracing::debug!("Worker IPC connect failed, retrying: {e}");
                std::thread::sleep(std::time::Duration::from_millis(50));
                continue;
            }
        };
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            continue; // empty read, retry
        }

        let cmd: WorkerCommand = match serde_json::from_str(line.trim()) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("Worker: failed to parse command: {e}");
                continue;
            }
        };

        match cmd {
            WorkerCommand::ForwardIntrospect { ref text, ref layers } => {
                let layer_set = layers.as_ref().map(|v| v.iter().copied().collect::<HashSet<_>>());
                let _ = model.forward_introspect_layers(text, layer_set);
            }
            WorkerCommand::Generate {
                ref text,
                max_new_tokens,
                temperature,
                top_p,
            } => {
                let _ = model.generate(text, max_new_tokens, temperature, top_p);
            }
            WorkerCommand::SetSteering { ref vectors } => {
                for (layer_idx, vec_data) in vectors {
                    if let Ok(t) = Tensor::new(vec_data.as_slice(), model.device())
                        .and_then(|t| t.to_dtype(DType::F32))
                    {
                        model.set_steering_vector(*layer_idx, t);
                    }
                }
            }
            WorkerCommand::SetSteeringRange {
                start_layer,
                end_layer,
                ref vector,
                scale,
            } => {
                if let Ok(t) = Tensor::new(vector.as_slice(), model.device())
                    .and_then(|t| t.to_dtype(DType::F32))
                {
                    let _ = model.set_steering_vectors_range(start_layer..end_layer, &t, scale);
                }
            }
            WorkerCommand::ClearSteering => {
                model.clear_steering_vectors();
            }
            WorkerCommand::SetPatch {
                layer_idx,
                ref hidden_state,
            } => {
                if let Ok(t) = Tensor::new(hidden_state.as_slice(), model.device()) {
                    model.set_patch(layer_idx, t);
                }
            }
            WorkerCommand::ClearPatches => {
                model.clear_patches();
            }
            WorkerCommand::SetCaptureRouting(enabled) => {
                model.set_capture_routing(enabled);
            }
            WorkerCommand::Shutdown => {
                tracing::info!("Worker received shutdown, exiting");
                return Ok(());
            }
        }
    }
}

// ── Master side (sends commands to workers) ─────────────────────────

/// Coordinator that sends commands to all TP workers before model operations.
pub struct WorkerCoordinator {
    #[cfg_attr(not(feature = "nccl"), allow(dead_code))]
    world_size: usize,
}

impl WorkerCoordinator {
    pub fn new(world_size: usize) -> Self {
        Self { world_size }
    }

    /// Send a command to all workers. Blocks until all workers have received it.
    #[cfg(feature = "nccl")]
    pub fn send_command(&self, cmd: &WorkerCommand) -> anyhow::Result<()> {
        use std::io::Write;

        use interprocess::local_socket::traits::Listener;
        use interprocess::local_socket::{GenericNamespaced, ListenerOptions, ToNsName};

        let ipc_name = CMD_SOCKET.to_ns_name::<GenericNamespaced>()?;
        let listener = ListenerOptions::new().name(ipc_name).create_sync()?;
        let payload = serde_json::to_string(cmd)? + "\n";

        let num_workers = self.world_size - 1;
        let mut sent = 0;
        while sent < num_workers {
            let mut stream = listener.accept()?;
            stream.write_all(payload.as_bytes())?;
            stream.flush()?;
            sent += 1;
        }
        Ok(())
    }

    /// No-op when NCCL is not compiled.
    #[cfg(not(feature = "nccl"))]
    pub fn send_command(&self, _cmd: &WorkerCommand) -> anyhow::Result<()> {
        Ok(())
    }
}
