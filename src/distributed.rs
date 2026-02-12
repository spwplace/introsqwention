//! Tensor parallelism setup for multi-GPU introspection.
//!
//! When the `nccl` feature is enabled, this module handles:
//! - Spawning worker processes (one per GPU, ranks 1..N-1)
//! - NCCL communicator initialization across all ranks
//! - IPC handshake so the master knows when workers are ready
//!
//! The master (rank 0) runs the Axum server; workers enter a command loop
//! that participates in NCCL collectives during forward passes.

use std::sync::Arc;

/// Configuration produced by the TP initialization phase.
pub struct TpConfig {
    pub rank: usize,
    #[allow(dead_code)]
    pub world_size: usize,
    pub comm: Option<Arc<mistralrs_quant::Comm>>,
}

impl TpConfig {
    /// Single-device configuration (Metal or single CUDA GPU).
    pub fn single() -> Self {
        Self {
            rank: 0,
            world_size: 1,
            comm: None,
        }
    }
}

// ── NCCL multi-GPU setup (only compiled with the `nccl` feature) ────

#[cfg(feature = "nccl")]
pub mod nccl_setup {
    use super::*;
    use core::ffi::c_char;
    use interprocess::local_socket::traits::{Listener, Stream};
    use interprocess::local_socket::{GenericNamespaced, ListenerOptions, Stream as LocalStream, ToNsName};
    use serde::{Deserialize, Serialize};
    use serde_big_array::BigArray;
    use std::io::{BufRead, BufReader, Write};

    const DAEMON_FLAG: &str = "__INTROSQWENTION_DAEMON";
    const IPC_SOCKET: &str = "introsqwention_daemon.sock";

    #[derive(Serialize, Deserialize)]
    struct WorkerInit {
        #[serde(with = "BigArray")]
        nccl_id: [c_char; 128],
        worker_rank: usize,
        world_size: usize,
    }

    /// Returns `true` if this process was spawned as a TP worker.
    pub fn is_worker() -> bool {
        std::env::var(DAEMON_FLAG).is_ok()
    }

    /// Master: create NCCL Id, spawn worker processes, wait for them,
    /// initialize rank-0 communicator.
    pub fn init_master(
        world_size: usize,
        device: &candle_core::Device,
    ) -> anyhow::Result<TpConfig> {
        assert!(
            world_size.is_power_of_two() && world_size >= 2,
            "TP world_size must be a power of 2 and >= 2, got {world_size}"
        );

        let id = mistralrs_quant::Id::new();
        let exe = std::env::current_exe()?;
        let args: Vec<String> = std::env::args().collect();

        // Spawn worker processes (ranks 1 .. world_size-1)
        for worker_rank in 0..(world_size - 1) {
            let init = WorkerInit {
                nccl_id: *id.internal(),
                worker_rank,
                world_size,
            };
            let mut cmd = std::process::Command::new(&exe);
            cmd.args(&args[1..]);
            cmd.env(DAEMON_FLAG, serde_json::to_string(&init)?);
            // Workers inherit stderr for tracing output
            cmd.stdout(std::process::Stdio::null());
            cmd.stderr(std::process::Stdio::inherit());
            cmd.stdin(std::process::Stdio::null());
            cmd.spawn()?;
        }

        // Wait for all workers to signal "ready" via IPC
        let ipc_name = IPC_SOCKET.to_ns_name::<GenericNamespaced>()?;
        let listener = ListenerOptions::new().name(ipc_name).create_sync()?;
        let mut ready_count = 0;
        while ready_count < world_size - 1 {
            let stream = listener.accept()?;
            let mut reader = BufReader::new(stream);
            let mut msg = String::new();
            reader.read_line(&mut msg)?;
            if msg.trim() == "ready" {
                ready_count += 1;
            }
        }
        tracing::info!("All {} TP workers ready", world_size - 1);

        // Initialize master's NCCL communicator (rank 0)
        let comm = mistralrs_quant::Comm::from_device(id, device, 0, world_size)?;
        Ok(TpConfig {
            rank: 0,
            world_size,
            comm: Some(Arc::new(comm)),
        })
    }

    /// Worker: read init payload from env, signal readiness to master,
    /// initialize this rank's NCCL communicator.
    pub fn init_worker() -> anyhow::Result<(TpConfig, WorkerMeta)> {
        let payload = std::env::var(DAEMON_FLAG)?;
        let init: WorkerInit = serde_json::from_str(&payload)?;
        let rank = init.worker_rank + 1; // master is rank 0

        let device = candle_core::Device::new_cuda(rank)?;

        // Signal master that we're ready
        let ipc_name = IPC_SOCKET.to_ns_name::<GenericNamespaced>()?;
        let mut stream = LocalStream::connect(ipc_name)?;
        stream.write_all(b"ready\n")?;
        drop(stream);

        tracing::info!("Worker rank {rank}/{} NCCL init", init.world_size);
        let comm = mistralrs_quant::Comm::from_device(
            mistralrs_quant::Id::uninit(init.nccl_id),
            &device,
            rank,
            init.world_size,
        )?;

        let tp = TpConfig {
            rank,
            world_size: init.world_size,
            comm: Some(Arc::new(comm)),
        };
        let meta = WorkerMeta { device };
        Ok((tp, meta))
    }

    /// Extra info returned to workers (device they should use).
    pub struct WorkerMeta {
        pub device: candle_core::Device,
    }
}
