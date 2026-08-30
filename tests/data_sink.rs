//! `read_into` and the abort path it exists for.

use unrar_ng::DataSink;
use unrar_ng::error::{Code, When};

/// The whole of `data/version.rar`'s only entry.
const VERSION: &[u8] = b"unrar-0.4.0";

fn first_entry() -> unrar_ng::OpenArchive<unrar_ng::Process, unrar_ng::CursorBeforeFile> {
    unrar_ng::Archive::new("data/version.rar")
        .open_for_processing()
        .expect("open")
        .read_header()
        .expect("read header")
        .expect("an entry")
}

/// Accepts everything, so `read_into` must behave exactly as `read`.
#[derive(Debug, Default)]
struct Accept(Vec<u8>);

impl DataSink for Accept {
    fn write_chunk(&mut self, chunk: &[u8]) -> bool {
        self.0.extend_from_slice(chunk);
        true
    }
}

/// Refuses once it has taken `limit` bytes, which is the case the trait exists for.
#[derive(Debug)]
struct Bounded {
    taken: Vec<u8>,
    limit: usize,
    refused: bool,
}

impl DataSink for Bounded {
    fn write_chunk(&mut self, chunk: &[u8]) -> bool {
        if self.taken.len() + chunk.len() > self.limit {
            self.refused = true;
            return false;
        }
        self.taken.extend_from_slice(chunk);
        true
    }
}

#[test]
fn read_into_an_accepting_sink_matches_read() {
    let (sink, result) = first_entry().read_into(Accept::default());
    assert!(result.is_ok(), "expected the archive back, got {result:?}");
    assert_eq!(sink.0, VERSION);
}

#[test]
fn read_still_returns_the_whole_entry() {
    // `read` is now a wrapper over `read_into`, so it is worth asserting it did not change.
    let (bytes, _archive) = first_entry().read().expect("read");
    assert_eq!(bytes, VERSION);
}

#[test]
fn a_sink_that_refuses_aborts_the_read() {
    let (sink, result) = first_entry().read_into(Bounded {
        taken: Vec::new(),
        limit: 0,
        refused: false,
    });

    // Not the DLL's `ERAR_UNKNOWN`: the wrapper knows it was the one that stopped.
    let error = result.expect_err("the sink refused, so the read cannot have completed");
    assert_eq!(error.code, Code::Aborted);
    assert_eq!(error.when, When::Process);

    // The sink comes back either way, holding what it took before it stopped.
    assert!(sink.refused);
    assert!(sink.taken.is_empty());
}

#[test]
fn a_bound_larger_than_the_entry_does_not_fire() {
    let (sink, result) = first_entry().read_into(Bounded {
        taken: Vec::new(),
        limit: VERSION.len(),
        refused: false,
    });

    assert!(result.is_ok(), "expected the archive back, got {result:?}");
    assert!(!sink.refused);
    assert_eq!(sink.taken, VERSION);
}

/// A solid, compressed entry takes the unpacker rather than the stored path, so the abort
/// is asserted against it too.
#[test]
fn the_abort_reaches_the_unpacker() {
    let entry = unrar_ng::Archive::new("data/solid.rar")
        .open_for_processing()
        .expect("open")
        .read_header()
        .expect("read header")
        .expect("an entry");

    let (sink, result) = entry.read_into(Bounded {
        taken: Vec::new(),
        limit: 0,
        refused: false,
    });

    assert_eq!(
        result.expect_err("the sink refused").code,
        Code::Aborted,
        "a compressed entry must abort the same way a stored one does"
    );
    assert!(sink.refused);
}
